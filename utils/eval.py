import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F

from utils.logging import AverageMeter, ProgressMeter, DistributedMeter
from utils.adv import pgd_whitebox, fgsm
from symbolic_interval.symbolic_network import (
    sym_interval_analyze,
    naive_interval_analyze,
    mix_interval_analyze,
)
from crown.bound_layers import (
    BoundSequential,
    BoundLinear,
    BoundConv2d,
    BoundDataParallel,
    Flatten,
)

from scipy.stats import norm
import numpy as np
import time


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def base(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "test-images",
                    torchvision.utils.make_grid(images[0: len(images) // 4]),
                )
        progress.display(i)  # print final results

    return top1.avg, top5.avg


def smooth(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc_1", ":6.2f") if not args.ddp else DistributedMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f") if not args.ddp else DistributedMeter("Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, top1, top5], prefix="Smooth (eval): "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # Default: evaluate on 10 random samples of additive gaussian noise.
            noise = torch.randn_like(images).to(device) * args.noise_std
            output = model(images + noise)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            # if i == 0 and writer:
            #     writer.add_image(
            #         "Adv-test-images",
            #         torchvision.utils.make_grid(images[0: len(images) // 4]),
            #     )

        if writer:
            progress.write_to_tensorboard(writer, "test", epoch * len(val_loader) + i)

        # write a sample of test images to tensorboard (helpful for debugging)
        if i == 0 and writer:
            writer.add_image(
                "Adv-test-images",
                torchvision.utils.make_grid(images[0: len(images) // 4]),
            )

    progress.display(i)  # print final results

    return top1.avg, np.nan
