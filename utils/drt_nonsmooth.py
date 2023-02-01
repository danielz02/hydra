from __future__ import annotations

import os
import sys
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.ensemble import Ensemble, Subspace
from utils import augmentations
from utils.eval import accuracy
from utils.logging import AverageMeter

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.utils_ensemble import get_stats, grad_l2, cosine_diversity


def drt_trainer_nonsmooth(
        args, train_loader: DataLoader, sm_loader: DataLoader, models: Union[Ensemble, Subspace], criterion,
        optimizer: Optimizer, epoch: int, noise_sd: float, device: torch.device, writer=None, train_sampler=None
):
    print(" ->->->->->->->->->-> Applying DRT w/o Smoothing <-<-<-<-<-<-<-<-<-<-")
    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")
    losses_lhs = AverageMeter("loss_lhs")
    losses_rhs = AverageMeter("loss_rhs")
    losses_con = AverageMeter("loss_con")
    losses_con_members = [AverageMeter(f"loss_con_{i}") for i in range(args.num_models)]

    losses_stab = AverageMeter("loss_stab")
    cosine = AverageMeter("Cosine", ":6.2f")
    l2 = AverageMeter("L2", ":6.2f")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    grad_l2_norm = AverageMeter("grad_l2")
    valid_pairs = AverageMeter("valid_pairs")

    end = time.time()

    models.train()
    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)
    if args.ddp:
        train_sampler.set_epoch(epoch)
        device = 'cuda'

    alphas = np.random.uniform(size=args.num_models)

    softma = nn.Softmax(1)
    for i, (inputs, targets) in enumerate(dataloader):
        if isinstance(inputs, list):
            images_all = torch.cat(inputs, 0).cuda()
            images_all.requires_grad = True
            images_all, targets = images_all.to(device), targets.to(device)
            batch_size = images_all.size(0)
        else:
            inputs.requires_grad = True
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            images_all = None

        if isinstance(models, Subspace) and args.sample_per_batch:
            alphas = np.random.uniform(size=args.num_models)

        loss_std = torch.tensor(0., device=device)

        if isinstance(models, Subspace):
            for j in range(args.num_models):
                models.fixed_alpha = alphas[j]
                logits = models(inputs)
                loss_std += criterion(logits, targets)
        else:
            for j in range(args.num_models):
                if args.augmix and not args.no_jsd:
                    targets = targets.cuda()
                    logits_all = models.models[j](images_all)
                    logits_clean, logits_aug1, logits_aug2 = torch.split(
                        logits_all, inputs[0].size(0)
                    )

                    # Cross-entropy is only computed on clean images
                    loss = F.cross_entropy(logits_clean, targets)

                    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1), \
                        F.softmax(logits_aug2, dim=1)

                    # Clamp mixture distribution to avoid exploding KL divergence
                    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                    loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

                    loss_std += loss
                    logits = logits_all
                else:
                    logits = models.models[j](inputs)
                    loss_std += criterion(logits, targets)

        rhsloss, rcount = torch.tensor(0., device=device), 0
        pred = []
        margin = []
        if args.augmix and not args.no_jsd:
            cur_input = images_all
            targets = targets.repeat(len(inputs))
        else:
            cur_input = inputs
        for j in range(args.num_models):
            if isinstance(models, Subspace):
                models.fixed_alpha = alphas[j]
                output = models(cur_input)
            else:
                output = models.models[j](cur_input)

            _, predicted = output.max(1)
            pred.append(predicted == targets)
            predicted = softma(output.sort()[0])
            predicted = predicted[:, -1] - predicted[:, -2]

            grad_outputs = torch.ones(predicted.shape)
            grad_outputs = grad_outputs.to(device)

            grad = torch.autograd.grad(predicted, cur_input, grad_outputs=grad_outputs,
                                       create_graph=True, only_inputs=True)[0]

            margin.append(grad.view(grad.size(0), -1))

            flg = pred[j].type(torch.FloatTensor).to(device)
            rhsloss += torch.sum(flg * predicted)
            rcount += torch.sum(flg)

        rhsloss /= max(rcount, 1)

        lhsloss, N = torch.tensor(0., device=device), torch.tensor(0., device=device)
        mse = nn.MSELoss(reduce=False)
        for ii in range(args.num_models):
            for j in range(ii + 1, args.num_models):
                flg = (pred[ii] & pred[j]).type(torch.FloatTensor).to(device)
                grad_norm = torch.sum(mse(margin[ii], -margin[j]), dim=1)
                lhsloss += torch.sum(grad_norm * flg)
                N += torch.sum(flg)
        lhsloss /= max(N, 1)

        loss = loss_std + args.lhs_weights * lhsloss - args.rhs_weights * rhsloss + args.beta

        if args.layer_type in ["curve", "line"] and args.beta_div and args.beta_div > 0:
            loss += cosine_diversity(models, args)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss_std.item(), batch_size)
        valid_pairs.update(N.item(), batch_size)
        losses_lhs.update(lhsloss.item(), batch_size)
        losses_rhs.update(rhsloss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        grad_l2_norm.update(grad_l2(models, device))
        if isinstance(models, Subspace):
            weight_cosine, weight_l2 = get_stats(models, args)
            cosine.update(weight_cosine, batch_size)
            l2.update(weight_l2, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Grad-L2 {grad_l2_norm.avg:.4f}\t'
                'Weight-L2 {l2.avg:.4f}\t'
                'Weight-Cosine {cosine.avg:.4f}\n'
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss-Std {loss.avg:.4f}\t'
                'Loss-Con {loss_con.avg:.4f}\t'
                'Loss-STAB {loss_stab.avg:.4f}\t'
                'Loss-GD {lhs.avg:.4f}\t'
                'Loss-CM {rhs.avg:.4f}\t'
                'Valid-Pairs {valid_pairs.avg:.4f}\t'
                'Acc@1 {top1.avg:.3f}\t'
                'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, l2=l2, cosine=cosine, grad_l2_norm=grad_l2_norm,
                    data_time=data_time, loss=losses, loss_stab=losses_stab, loss_con=losses_con, lhs=losses_lhs,
                    rhs=losses_rhs, valid_pairs=valid_pairs, top1=top1, top5=top5
                )
            )

    if writer:
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('loss/lhs', losses_lhs.avg, epoch)
        writer.add_scalar('loss/rhs', losses_rhs.avg, epoch)
        writer.add_scalar('loss/con', losses_con.avg, epoch)
        writer.add_scalar('loss/stab', losses_stab.avg, epoch)
        writer.add_scalar('loss/valid_pairs', valid_pairs.avg, epoch)
        [writer.add_scalar(f'loss/con_{i}', meter.avg, epoch) for i, meter in enumerate(losses_con_members)]
        writer.add_scalar('weight/l2', l2.avg, epoch)
        writer.add_scalar('weight/cosine', cosine.avg, epoch)
        writer.add_scalar('grad/l2', grad_l2_norm.avg, epoch)
        writer.add_scalar('batch_time', batch_time.avg, epoch)
        writer.add_scalar('accuracy/train@1', top1.avg, epoch)
        writer.add_scalar('accuracy/train@5', top5.avg, epoch)

        if isinstance(models, Subspace):
            for i, alpha in enumerate(alphas):
                writer.add_scalar(f'alpha/{i}', alpha, epoch)
