import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from data.imagenet import ImageFolderLMDB


class TinyImageNet:
    def __init__(self, args):
        self.args = args
        self.tr_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.tr_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_loader, self.val_loader = None, None

    def data_loaders(self):
        # train_set = ImageFolderLMDB(os.path.join(self.args.data_dir, "train.lmdb"), self.tr_train)
        # val_set = ImageFolderLMDB(os.path.join(self.args.data_dir, "val.lmdb"), self.tr_test)
        train_set = ImageFolder(os.path.join(self.args.data_dir, "train"), self.tr_train)
        val_set = ImageFolder(os.path.join(self.args.data_dir, "val"), self.tr_test)

        train_loader = DataLoader(
            train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_set, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        return train_loader, val_loader
