from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.ensemble import Ensemble, Subspace
from utils.consistency import consistency_loss
from utils.eval import accuracy
from utils.logging import DistributedMeter, AverageMeter

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.smoothadv import Attacker
from utils.utils_ensemble import requires_grad_, get_stats, grad_l2, stability_loss, cosine_diversity


def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i * batch_size: (i + 1) * batch_size], y[i * batch_size: (i + 1) * batch_size]


def DRT_Trainer(
        args, train_loader: DataLoader, sm_loader: DataLoader, models: Ensemble | Subspace, criterion,
        optimizer: Optimizer, epoch: int, noise_sd: float, attacker: Attacker, device: torch.device, writer=None,
        train_sampler=None
):
    if args.ddp:
        import horovod.torch as hvd
        train_sampler.set_epoch(epoch)
        device = 'cuda'
        is_rank0 = (args.ddp and hvd.rank() == 0) or not args.ddp
    else:
        is_rank0 = True

    if is_rank0:
        print(" ->->->->->->->->->-> One epoch with DRT <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")
    losses_lhs = AverageMeter("loss_lhs")  # if not args.ddp else DistributedMeter("loss_lhs")
    losses_rhs = AverageMeter("loss_rhs")
    losses_con = AverageMeter("loss_con")
    losses_con_members = [AverageMeter(f"loss_con_{i}") for i in range(args.num_models)]
    losses_stab = AverageMeter("loss_stab")
    cosine = AverageMeter("Cosine", ":6.2f")
    l2 = AverageMeter("L2", ":6.2f")
    top1 = AverageMeter("top1")  # if not args.ddp else DistributedMeter("Acc_1")
    top5 = AverageMeter("top5")  # if not args.ddp else DistributedMeter("Acc_5")
    grad_l2_norm = AverageMeter("grad_l2")
    valid_pairs = AverageMeter("valid_pairs")  # if not args.ddp else DistributedMeter("valid_pairs")

    end = time.time()

    models.train()
    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

    alphas = np.random.uniform(size=args.num_models)

    softma = nn.Softmax(1)
    for i, batch in enumerate(dataloader):
        data_time.update(time.time() - end)

        if args.is_semisup:
            batch = (
                torch.cat([d[0] for d in batch], 0),
                torch.cat([d[1] for d in batch], 0),
            )

        if i == 0 and is_rank0:
            print(
                batch[0].shape,
                batch[1].shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(batch[0]).data.cpu().numpy(),
                    torch.max(batch[0]).data.cpu().numpy(),
                )
            )

        mini_batches = get_minibatches(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            noises = [torch.randn_like(inputs, device=device) * noise_sd
                      for _ in range(args.num_noise_vec)]

            if isinstance(models, Subspace) and args.sample_per_batch:
                alphas = np.random.uniform(size=args.num_models)

            if args.adv_training:
                if isinstance(models, Subspace):
                    raise NotImplementedError("SmoothAdv is not supported with self-ensemble!")
                adv_x = []
                for j in range(args.num_models):
                    requires_grad_(models.models[j], False)
                    models.models[j].eval()
                    adv = attacker.attack(models.models[j], inputs, targets, noises=noises)
                    models.models[j].train()
                    requires_grad_(models.models[j], True)
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
            loss_std = torch.tensor(0., device=device)
            loss_stab = torch.tensor(0., device=device)
            loss_consistency = torch.tensor(0., device=device)

            if isinstance(models, Subspace):
                for j in range(args.num_models):
                    models.fixed_alpha = alphas[j]
                    if args.adv_training:
                        logits = models(adv_input[j])
                    else:
                        logits = models(noisy_input)
                    loss_std += criterion(logits, targets)
                    if args.drt_stab:
                        clean_input = torch.cat([inputs for _ in noises], dim=0)
                        loss_stab += stability_loss(noisy_logits=logits, clean_logits=models(clean_input))
                    elif args.drt_consistency:
                        logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
                        loss_consistency += consistency_loss(logits_chunk, lbd=args.lbd)
            else:
                for j in range(args.num_models):
                    if args.adv_training:
                        logits = models.models[j](adv_input[j])
                    else:
                        logits = models.models[j](noisy_input)
                    loss_std += criterion(logits, targets)
                    if args.drt_stab:
                        clean_input = torch.cat([inputs for _ in noises], dim=0)
                        loss_stab += stability_loss(noisy_logits=logits, clean_logits=models.models[j](clean_input))
                    elif args.drt_consistency:
                        logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
                        loss_consistency += consistency_loss(logits_chunk, lbd=args.lbd)

            rhsloss, rcount = torch.tensor(0., device=device), 0
            pred = []
            margin = []
            for j in range(args.num_models):
                if args.adv_training:
                    cur_input = adv_input[j]
                else:
                    cur_input = noisy_input
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
            mse = nn.MSELoss(reduction='none')
            for ii in range(args.num_models):
                for j in range(ii + 1, args.num_models):
                    flg = (pred[ii] & pred[j]).type(torch.FloatTensor).to(device)
                    grad_norm = torch.sum(mse(margin[ii], -margin[j]), dim=1)
                    lhsloss += torch.sum(grad_norm * flg)
                    N += torch.sum(flg)

            lhsloss /= max(N, 1)

            loss = loss_std + args.lhs_weights * lhsloss - args.rhs_weights * rhsloss + args.beta * loss_stab + \
                loss_consistency

            if args.layer_type in ["curve", "line"] and args.beta_div and args.beta_div > 0:
                loss += cosine_diversity(models, args)

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss_std, batch_size)
            valid_pairs.update(N, batch_size)
            losses_lhs.update(lhsloss, batch_size)
            losses_rhs.update(rhsloss, batch_size)
            losses_stab.update(loss_stab, batch_size)
            losses_con.update(loss_consistency, batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

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

        if i % args.print_freq == 0 and is_rank0:
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
        writer.add_scalar('loss/stab', losses_stab.avg, epoch)
        writer.add_scalar('loss/valid_pairs', valid_pairs.avg, epoch)
        [writer.add_scalar(f'loss/con_{i}', meter.avg, epoch) for i, meter in enumerate(losses_con_members)]
        writer.add_scalar('loss/con', losses_con.avg, epoch)
        writer.add_scalar('weight/l2', l2.avg, epoch)
        writer.add_scalar('weight/cosine', cosine.avg, epoch)
        writer.add_scalar('grad/l2', grad_l2_norm.avg, epoch)
        writer.add_scalar('batch_time', batch_time.avg, epoch)
        writer.add_scalar('accuracy/train@1', top1.avg, epoch)
        writer.add_scalar('accuracy/train@5', top5.avg, epoch)

        if isinstance(models, Subspace):
            for i, alpha in enumerate(alphas):
                writer.add_scalar(f'alpha/{i}', alpha, epoch)
