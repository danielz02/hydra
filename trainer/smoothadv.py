import time

import numpy as np
import torch

from utils.eval import accuracy
from models.ensemble import Ensemble
from utils.drt import get_minibatches
from utils.logging import AverageMeter

# TODO: support sm_loader when len(sm_loader.dataset) < len(train_loader.dataset)
from utils.model import prepare_model
from utils.smoothadv import PGD_L2, DDN
from utils.utils_ensemble import requires_grad_


def train(models: Ensemble, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer):
    print(" ->->->->->->->->->-> One epoch with SmoothAdv <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    end = time.time()

    if args.attack == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device=device, max_norm=args.epsilon)
    elif args.attack == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(
            steps=args.num_steps, device=device, max_norm=args.epsilon, init_norm=args.init_norm_DDN,
            gamma=args.gamma_DDN
        )
    else:
        raise Exception('Unknown attack')
    attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon / args.warmup])
    attacker.init_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon / args.warmup])

    for i in range(args.num_models):
        models.models[i].train()

    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

    for i, batch in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.is_semisup:
            batch = (
                torch.cat([d[0] for d in batch], 0),
                torch.cat([d[1] for d in batch], 0),
            )

        mini_batches = get_minibatches(batch, args.num_noise_vec)

        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.repeat((1, args.num_noise_vec, 1, 1)).reshape(-1, *batch[0].shape[1:])
            adv_x = []
            batch_size = inputs.size(0)
            noise = torch.randn_like(inputs, device=device) * args.noise_std
            for j in range(args.num_models):
                requires_grad_(models.models[j], False)
                models.models[j].eval()
                adv = attacker.attack(
                    models.models[j], inputs, targets, noise=noise, num_noise_vectors=args.num_noise_vec,
                    no_grad=args.no_grad_attack
                )
                models.models[j].train()
                prepare_model(models.models, args)
                adv_x.append(adv)

            adv_input = []
            for j in range(args.num_models):
                noisy_input = torch.cat([adv_x[j] + noise], dim=0)
                adv_input.append(noisy_input)

            # augment inputs with noise

            targets = targets.unsqueeze(1).repeat(1, args.num_noise_vec).reshape(-1, 1).squeeze()

            loss_std = 0

            assert args.num_models >= 1
            for j in range(args.num_models):
                logits = models.models[j](adv_input[j])
                loss_std += criterion(logits, targets)

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss_std.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_std.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {loss.avg:.4f}\t'
                'Acc@1 {top1.avg:.3f}\t'
                'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5
                )
            )

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    writer.add_scalar('accuracy/train@5', top5.avg, epoch)
