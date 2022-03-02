from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
from torch.distributions.normal import Normal
import torch.nn.functional as F

import sys
import os

from models.ensemble import Ensemble, BezierCurve
from utils.model import prepare_model

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.smoothadv import Attacker
from utils.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_, get_stats


def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i * batch_size: (i + 1) * batch_size], y[i * batch_size: (i + 1) * batch_size]


def DRT_Trainer(
        args, train_loader: DataLoader, sm_loader: DataLoader, models: Ensemble | BezierCurve, criterion,
        optimizer: Optimizer, epoch: int, noise_sd: float, attacker: Attacker, device: torch.device, writer=None
):
    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")
    losses_lhs = AverageMeter("loss_lhs")
    losses_rhs = AverageMeter("loss_rhs")
    cosine = AverageMeter("Cosine", ":6.2f")
    l2 = AverageMeter("L2", ":6.2f")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")

    end = time.time()

    models.train()
    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

    softma = nn.Softmax(1)
    for i, batch in enumerate(dataloader):
        data_time.update(time.time() - end)

        if args.is_semisup:
            batch = (
                torch.cat([d[0] for d in batch], 0),
                torch.cat([d[1] for d in batch], 0),
            )

        # TODO: Bug fix - use same sample of models in  benign loss and regularizer
        mini_batches = get_minibatches(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            noises = [torch.randn_like(inputs, device=device) * noise_sd
                      for _ in range(args.num_noise_vec)]

            if args.adv_training:
                if isinstance(models, BezierCurve):
                    raise NotImplementedError("SmoothAdv is not supported with self-ensemble!")
                adv_x = []
                for j in range(args.num_models):
                    requires_grad_(models.models[j], False)
                    models.models[j].eval()
                    adv = attacker.attack(models.models[j], inputs, targets, noises=noises)
                    models.models[j].train()
                    prepare_model(models.models[j], args)
                    adv_x.append(adv)

                adv_input = []
                for j in range(args.num_models):
                    noisy_input = torch.cat([adv_x[j] + noise for noise in noises], dim=0)
                    noisy_input.requires_grad = True
                    adv_input.append(noisy_input)
            else:
                noisy_input = torch.cat([inputs + noise for noise in noises], dim=0)
                noisy_input.requires_grad = True

            targets = targets.repeat(args.num_noise_vec)
            loss_std = 0

            if isinstance(models, BezierCurve):
                logits = models(noisy_input)
                loss_std = criterion(logits, targets)
            else:
                for j in range(args.num_models):
                    if args.adv_training:
                        logits = models.models[j](adv_input[j])
                    else:
                        logits = models.models[j](noisy_input)
                    loss_std += criterion(logits, targets)

            rhsloss, rcount = 0, 0
            pred = []
            margin = []
            for j in range(args.num_models):
                if args.adv_training:
                    cur_input = adv_input[j]
                else:
                    cur_input = noisy_input
                if isinstance(models, BezierCurve):
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

            lhsloss, N = 0, 0
            mse = nn.MSELoss(reduce=False)
            for ii in range(args.num_models):
                for j in range(ii + 1, args.num_models):
                    flg = (pred[ii] & pred[j]).type(torch.FloatTensor).to(device)
                    grad_norm = torch.sum(mse(margin[ii], -margin[j]), dim=1)
                    lhsloss += torch.sum(grad_norm * flg)
                    N += torch.sum(flg)

            lhsloss /= max(N, 1)

            losses_lhs.update(lhsloss.item(), batch_size)

            loss = loss_std + args.lhs_weights * lhsloss - args.rhs_weights * rhsloss

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss_std.item(), batch_size)
            losses_rhs.update(rhsloss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            if args.layer_type == "curve":
                weight_cosine, weight_l2 = get_stats(models, args)
                cosine.update(weight_cosine, batch_size)
                l2.update(weight_l2, batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Weight-L2 {l2.avg:.4f}\t'
                'Weight-Cosine {cosine.avg:.4f}\t'
                'Loss-Std {loss.avg:.4f}\t'
                'Loss-GD {lhs.avg:.4f}\t'
                'Loss-CM {rhs.avg:.4f}\t'
                'Acc@1 {top1.avg:.3f}\t'
                'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, l2=l2, cosine=cosine,
                    data_time=data_time, loss=losses, lhs=losses_lhs, rhs=losses_rhs, top1=top1, top5=top5
                )
            )

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('loss/lhs', losses_lhs.avg, epoch)
    writer.add_scalar('loss/rhs', losses_rhs.avg, epoch)
    writer.add_scalar('weight/l2', l2.avg, epoch)
    writer.add_scalar('weight/cosine', cosine.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    writer.add_scalar('accuracy/train@5', top5.avg, epoch)
