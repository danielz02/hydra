from utils.trs import TRS_Trainer


def train(model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer):
    print(" ->->->->->->->->->-> One epoch with TRS <-<-<-<-<-<-<-<-<-<-")

    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)
    TRS_Trainer(args, dataloader, False, model, criterion, optimizer, epoch, device, writer)
