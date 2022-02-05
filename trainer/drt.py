from utils.drt import DRT_Trainer
from utils.smoothadv import SmoothAdv_PGD


def train(model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer):
    print(" ->->->->->->->->->-> One epoch with DRT <-<-<-<-<-<-<-<-<-<-")

    if args.adv_training:
        attacker = SmoothAdv_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon)
    else:
        attacker = None

    DRT_Trainer(args, train_loader, sm_loader, model, criterion, optimizer, epoch, args.noise_std, attacker, device, writer)
