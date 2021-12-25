import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torchvision
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_
from models.ensemble import Ensemble
from utils.utils_ensemble import Cosine, Magnitude
from utils.distillation import Linf_distillation


def PGD(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.

    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).cuda()
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        # adv.requires_grad_()
        grad_loss = 0
        for i, m in enumerate(models.models):
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        grad_loss /= 3
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()


def TRS_Trainer(args, loader: DataLoader, sm_loader: bool, models: Ensemble, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    cos_losses = AverageMeter("cos/losses")
    smooth_losses = AverageMeter("smooth_losses")
    cos01_losses = AverageMeter("cos01/losses")
    cos02_losses = AverageMeter("cos02/losses")
    cos12_losses = AverageMeter("cos12/losses")

    end = time.time()

    for i in range(args.num_models):
        models.models[i].train()
        requires_grad_(models.models[i], True)

    for i, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if sm_loader:
            inputs, targets = (
                torch.cat([d[0] for d in data], 0).to(device),
                torch.cat([d[1] for d in data], 0).to(device),
            )
        else:
            inputs, targets = data[0].to(device), data[1].to(device)

        # basic properties of training data
        if i == 0:
            print(
                inputs.shape,
                targets.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(f"Training images range: {[torch.min(inputs), torch.max(inputs)]}")

        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models.models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        cos_loss, smooth_loss = 0, 0

        cos01 = Cosine(grads[0], grads[1])
        cos02 = Cosine(grads[0], grads[2])
        cos12 = Cosine(grads[1], grads[2])

        cos_loss = (cos01 + cos02 + cos12) / 3.

        N = inputs.shape[0] // 2
        cureps = (args.adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
        clean_inputs = inputs[:N].detach()  # PGD(self.models, inputs[:N], targets[:N])
        adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()

        adv_x = torch.cat([clean_inputs, adv_inputs])

        adv_x.requires_grad = True

        if args.plus_adv:
            for j in range(args.num_models):
                outputs = models.models[j](adv_x)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        else:
            # grads = []
            for j in range(args.num_models):
                outputs = models.models[j](inputs)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        smooth_loss /= 3

        loss = loss_std + args.scale * (args.coeff * cos_loss + args.lamda * smooth_loss)

        logits = models(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        cos_losses.update(cos_loss.item(), batch_size)
        smooth_losses.update(smooth_loss.item(), batch_size)
        cos01_losses.update(cos01.item(), batch_size)
        cos02_losses.update(cos02.item(), batch_size)
        cos12_losses.update(cos12.item(), batch_size)

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
                'Loss {loss.avg:.4f}\t'
                'Acc@1 {top1.avg:.3f}\t'
                'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5
                )
            )

        # write a sample of training images to tensorboard (helpful for debugging)
        if i == 0:
            writer.add_image(
                "training-images",
                torchvision.utils.make_grid(inputs[0: len(inputs) // 4]),
            )

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/cos_loss', cos_losses.avg, epoch)
    writer.add_scalar('train/smooth_loss', smooth_losses.avg, epoch)
    writer.add_scalar('train/cos01', cos01_losses.avg, epoch)
    writer.add_scalar('train/cos02', cos02_losses.avg, epoch)
    writer.add_scalar('train/cos12', cos12_losses.avg, epoch)
