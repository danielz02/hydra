from utils.drt import DRT_Trainer
from utils.smoothadv import SmoothAdv_PGD


def train(model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer):
    print(" ->->->->->->->->->-> One epoch with DR <-<-<-<-<-<-<-<-<-<-")

    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)
    if args.adv_training:
        attacker = SmoothAdv_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon)
    else:
        attacker = None

    DRT_Trainer(args, dataloader, model, criterion, optimizer, epoch, args.noise_std, attacker, device, writer)
