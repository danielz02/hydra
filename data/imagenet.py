import os
import numpy as np
import six
import lmdb
import torch
import pickle
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler


# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used along with in the training and eval.
class imagenet:
    """ 
        imagenet dataset.
    """

    def __init__(self, args):
        self.args = args

        self.tr_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = ImageFolderLMDB(
            os.path.join(self.args.data_dir, "train.lmdb"), self.tr_train
        )
        testset = ImageFolderLMDB(
            os.path.join(self.args.data_dir, "val.lmdb"), self.tr_test
        )

        if self.args.ddp:
            import horovod.torch as hvd
            import torch.multiprocessing as mp

            kwargs = {'num_workers': 1, 'pin_memory': True}
            # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
            # issues with Infiniband implementations that are not fork-safe
            if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                    mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
                kwargs['multiprocessing_context'] = 'forkserver'

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=self.args.batch_size, sampler=train_sampler, **kwargs
            )
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=self.args.test_batch_size, sampler=test_sampler, **kwargs
            )
        else:
            train_loader = DataLoader(
                trainset,
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=16,
                pin_memory=True,
                **kwargs,
            )

            test_loader = DataLoader(
                testset,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                **kwargs,
            )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)
