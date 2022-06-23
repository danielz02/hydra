import time
import torch

from models.ensemble import Subspace
from utils.consistency import consistency_loss
from utils.drt import get_minibatches
from utils.eval import accuracy
from utils.logging import AverageMeter
from utils.smoothadv import SmoothAdv_PGD
from utils.utils_ensemble import requires_grad_, grad_l2, get_stats, cosine_diversity, subspace_diversity


def train(model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer, train_sampler):
    if args.ddp:
        import horovod.torch as hvd
        train_sampler.set_epoch(epoch)
        device = 'cuda'
        is_rank0 = (args.ddp and hvd.rank() == 0) or not args.ddp
    else:
        is_rank0 = True

    if is_rank0:
        print(" ->->->->->->->->->-> One epoch with Consistency <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")
    losses_con = AverageMeter("loss_con")
    losses_gd = AverageMeter("loss_gd")
    cosine = AverageMeter("Cosine", ":6.2f")
    l2 = AverageMeter("L2", ":6.2f")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    grad_l2_norm = AverageMeter("grad_l2")
    end = time.time()

    if args.adv_training:
        attacker = SmoothAdv_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon)
    else:
        attacker = None

    # switch to train mode
    model.train()

    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

    for i, batch in enumerate(dataloader):
        # measure data loading time
        if args.is_semisup:
            batch = (
                torch.cat([d[0] for d in batch], 0),
                torch.cat([d[1] for d in batch], 0),
            )
        data_time.update(time.time() - end)

        mini_batches = get_minibatches(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            noises = [torch.randn_like(inputs, device=device) * args.noise_std
                      for _ in range(args.num_noise_vec)]

            if args.adv_training:
                requires_grad_(model, False)
                model.eval()
                inputs = attacker.attack(model, inputs, targets, noises=noises)
                model.train()

            # augment inputs with noise
            inputs_c = torch.cat([inputs + noise for noise in noises], dim=0)
            targets_c = targets.repeat(args.num_noise_vec)

            logits = model(inputs_c)
            loss_xent = criterion(logits, targets_c)

            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            loss_con = consistency_loss(logits_chunk, args.lbd)

            loss = loss_xent + loss_con

            if args.layer_type in ["curve", "line"] and args.beta_div and args.beta_div > 0:
                loss += cosine_diversity(model, args)

            acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_con.update(loss_con.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_gd = subspace_diversity(model, (inputs_c, targets_c), None, device, args)
            losses_gd.update(loss_gd, batch_size)

            grad_l2_norm.update(grad_l2(model, device))
            if isinstance(model, Subspace):
                weight_cosine, weight_l2 = get_stats(model, args)
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
                'Loss-GD {gd.avg:.4f}\t'
                'Loss-Std {loss.avg:.4f}\t'
                'Loss-Con {loss_con.avg:.4f}\t'
                'Grad-L2 {grad_l2_norm.avg:.4f}\t'
                'Weight-L2 {l2.avg:.4f}\t'
                'Weight-Cosine {cosine.avg:.4f}\t'
                'Acc@1 {top1.avg:.3f}\t'
                'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, l2=l2, cosine=cosine, grad_l2_norm=grad_l2_norm,
                    data_time=data_time, loss=losses, loss_con=losses_con, top1=top1, top5=top5, gd=losses_gd
                )
            )

    if writer:
        writer.add_scalar('loss/gd', losses_gd.avg, epoch)
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('loss/consistency', losses_con.avg, epoch)
        writer.add_scalar('weight/l2', l2.avg, epoch)
        writer.add_scalar('weight/cosine', cosine.avg, epoch)
        writer.add_scalar('grad/l2', grad_l2_norm.avg, epoch)
        writer.add_scalar('batch_time', batch_time.avg, epoch)
        writer.add_scalar('accuracy/train@1', top1.avg, epoch)
        writer.add_scalar('accuracy/train@5', top5.avg, epoch)
