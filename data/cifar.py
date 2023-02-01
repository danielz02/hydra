import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import augmentations


# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used along with in the training and eval.
class CIFAR10:
    """ 
        CIFAR-10 dataset.
    """

    def __init__(self, args):
        assert not (args.normalize and args.is_semisup)

        self.args = args

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        if self.args.augmix:
            trainset = datasets.CIFAR10(
                root=os.path.join(self.args.data_dir, "CIFAR10"),
                train=True,
                download=True,
            )
            trainset = AugMixDataset(trainset, self.tr_test, self.args, self.args.no_jsd)
        else:
            trainset = datasets.CIFAR10(
                root=os.path.join(self.args.data_dir, "CIFAR10"),
                train=True,
                download=True,
                transform=self.tr_train,
            )

        # subset_indices = np.random.permutation(np.arange(len(trainset)))[
        #                  : int(self.args.data_fraction * len(trainset))
        #                  ]
        # if self.args.data_fraction is not None:
        #     raise NotImplementedError

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            # sampler=SubsetRandomSampler(subset_indices),
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, pin_memory=True,
            num_workers=4, **kwargs
        )

        print(
            f"Traing loader: {len(trainset)} images, Test loader: {len(testset)} images"
        )
        return train_loader, test_loader


class CIFAR100:
    """ 
        CIFAR-100 dataset.
    """

    def __init__(self, args):
        self.args = args

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        # subset_indices = np.random.permutation(np.arange(len(trainset)))[
        #                  : int(self.args.data_fraction * len(trainset))
        #                  ]
        if self.args.data_fraction is not None:
            raise NotImplementedError

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            # sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


def aug(image, preprocess, args):
    """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    args: results from argument parser
  Returns:
    mixed: Augmented and mixed image.
  """
    aug_list = augmentations.augmentations
    if args.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, args, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.args = args

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess, self.args), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.args),
                        aug(x, self.preprocess, self.args))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)
