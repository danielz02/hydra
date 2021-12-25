import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class MNIST:
    def __init__(self, args, normalize=False):
        self.args = args
        self.tr_train = [
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(transforms.Normalize(0.5, 0.5))
            self.tr_test.append(transforms.Normalize(0.5, 0.5))

    def data_loaders(self, **kwargs):
        train_set = datasets.MNIST(
            os.path.join(self.args.data_dir, "MNIST"), train=True, download=True,
            transform=transforms.Compose(self.tr_train)
        )
        test_set = datasets.MNIST(
            os.path.join(self.args.data_dir, "MNIST"), train=False, transform=transforms.Compose(self.tr_test)
        )
        train_loader = DataLoader(
            train_set, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count(),
            **kwargs
        )
        test_loader = DataLoader(
            test_set, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count(),
            **kwargs
        )

        return train_loader, test_loader
