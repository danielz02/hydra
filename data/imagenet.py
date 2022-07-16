import os
import numpy as np
import six
import lmdb
import torch
import pickle
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
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

        self.train_sampler = None

    def data_loaders(self, **kwargs):
        trainset = ImageFolderLMDB(
            os.path.join(self.args.data_dir, "train.lmdb"), self.tr_train
        )
        testset = ImageFolderLMDB(
            os.path.join(self.args.data_dir, "val.lmdb"), self.tr_test
        )

        # trainset = ImageFolder(
        #     os.path.join(self.args.data_dir, "train"), self.tr_train
        # )
        # testset = ImageFolder(
        #     os.path.join(self.args.data_dir, "val"), self.tr_test
        # )
        print(trainset, testset)

        if self.args.ddp:
            import horovod.torch as hvd
            import torch.multiprocessing as mp

            kwargs = {'num_workers': 4, 'pin_memory': True}
            # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
            # issues with Infiniband implementations that are not fork-safe
            if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                    mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
                kwargs['multiprocessing_context'] = 'forkserver'

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
            )
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=self.args.batch_size, sampler=train_sampler, persistent_workers=True, timeout=600,
                **kwargs
            )
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=self.args.test_batch_size, sampler=test_sampler, persistent_workers=True,
                timeout=600, **kwargs
            )
            self.train_sampler = train_sampler
        else:
            train_loader = DataLoader(
                trainset,
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=8,
                pin_memory=True,
                **kwargs,
            )

            test_loader = DataLoader(
                testset,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                **kwargs,
            )

        print(
            f"Training loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = None
        self.txn = None

        self.transform = transform
        self.target_transform = target_transform

        env = lmdb.open(
            self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        with env.begin(write=False, buffers=True) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        env.close()

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        self.txn = self.env.begin(write=False, buffers=True)

    def __getitem__(self, index):
        if self.txn is None:
            self._init_db()

        # with self.env.begin(write=False) as txn:
        byteflow = self.txn.get(self.keys[index])
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

        # im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return im2arr, target

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
