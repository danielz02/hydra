import time

import torch
import torchvision
import torch.nn as nn

from utils.eval import accuracy
from models.ensemble import Ensemble, Subspace
from utils.logging import AverageMeter, ProgressMeter

# TODO: support sm_loader when len(sm_loader.dataset) < len(train_loader.dataset)
from utils.utils_ensemble import cosine_diversity, get_stats


def train(
        model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer, train_sampler=None
):
    if args.ddp:
        import horovod.torch as hvd

        train_sampler.set_epoch(epoch)
        device = 'cuda'
        is_rank0 = (args.ddp and hvd.rank() == 0) or not args.ddp
    else:
        is_rank0 = True

    if is_rank0:
        print(" ->->->->->->->->->-> One epoch with Gaussian Smoothing <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    cosine = AverageMeter("Cosine", ":6.2f")
    l2 = AverageMeter("L2", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, cosine, l2, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    if isinstance(model, Ensemble):
        for m in model.models:
            m.train()

    end = time.time()

    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

    for i, data in enumerate(dataloader):
        if sm_loader:
            images, target = (
                torch.cat([d[0] for d in data], 0).to(device),
                torch.cat([d[1] for d in data], 0).to(device),
            )
        else:
            images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0 and is_rank0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        noise = torch.randn_like(images, device=device) * args.noise_std
        if isinstance(model, Ensemble) and not isinstance(model, Subspace):
            loss = []
            output_ensemble = 0
            for m in model.models:
                output = m(images + noise)
                output_ensemble += output
                loss.append(nn.CrossEntropyLoss()(output, target))
            output = output_ensemble / len(model.models)
        else:
            output = model(images + noise)
            loss = nn.CrossEntropyLoss()(output, target)

        if args.layer_type in ["curve", "line"] and args.beta_div and args.beta_div > 0:
            loss += cosine_diversity(model, args)

        optimizer.zero_grad()
        if isinstance(loss, list):
            for sub_loss in loss:
                sub_loss.backward()
            loss = sum(loss)
        else:
            loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if args.layer_type in ["line", "curve"]:
            weight_cosine, weight_l2 = get_stats(model, args)
            cosine.update(weight_cosine, images.size(0))
            l2.update(weight_l2, images.size(0))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_rank0:
            progress.display(i)
            progress.write_to_tensorboard(
                writer, "train", epoch * len(train_loader) + i
            )

        # write a sample of training images to tensorboard (helpful for debugging)
        if i == 0 and is_rank0:
            writer.add_image(
                "training-images",
                torchvision.utils.make_grid(images[0: len(images) // 4]),
            )
