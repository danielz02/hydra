from utils.drt import DRT_Trainer
from utils.drt_semisup import DRT_Trainer_Separate_Semisup
from utils.smoothadv import SmoothAdv_PGD


def train(model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer, train_sampler=None):

    if args.adv_training:
        attacker = SmoothAdv_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon)
    else:
        attacker = None

    if args.separate_semisup:
        DRT_Trainer_Separate_Semisup(
            args, train_loader, sm_loader, model, criterion, optimizer, epoch, args.noise_std, attacker, device, writer,
            train_sampler
        )
    else:
        DRT_Trainer(
            args, train_loader, sm_loader, model, criterion, optimizer, epoch, args.noise_std, attacker, device, writer,
            train_sampler
        )
