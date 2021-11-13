import os

import numpy as np
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .tar_dataset import TarImageFolder

data_root = "datasets/"
cifar10c = os.path.join(data_root, "CIFAR-10-C")
cifar100c = os.path.join(data_root, "CIFAR-100-C")
imagenetc = os.path.join(data_root, "ImageNet-C")
imagenetval = os.path.join(data_root, "ImageNetVal")


imagenet_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

cifar_transform = transforms.Compose([transforms.ToTensor()])


class CIFAR10_C(torchvision.datasets.CIFAR10):
    def __init__(self, corruption, level, transform=None, target_transform=None):
        tesize = 10000
        super().__init__(
            root=cifar10c, train=False, download=False, transform=transform, target_transform=target_transform
        )
        teset_raw = np.load(os.path.join(cifar10c, f"{corruption}.npy"))
        self.data = teset_raw[(level - 1) * tesize : level * tesize]


class CIFAR100_C(torchvision.datasets.CIFAR100):
    def __init__(self, corruption, level, transform=None, target_transform=None):
        tesize = 10000
        super().__init__(
            root=cifar100c, train=False, download=False, transform=transform, target_transform=target_transform
        )
        teset_raw = np.load(os.path.join(cifar100c, f"{corruption}.npy"))
        self.data = teset_raw[(level - 1) * tesize : level * tesize]


class ImagenetC(TarImageFolder):
    def __init__(self, corruption, level):
        tarfile = os.path.join(imagenetc, "{}-{}.tar".format(corruption, level))
        super().__init__(tarfile, transform=None, root_in_archive="{}".format(level))()


class MyDummyDataset(Dataset):
    def __init__(self):
        self.ds = None

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]


class Imagenet(MyDummyDataset):
    def __init__(self, corruption, level, transform=None):
        if corruption == 'original':
            self.ds = torchvision.datasets.ImageFolder(imagenetval, transform=transform)
        else:
            self.ds = ImagenetC(corruption=corruption, level=level)


class CIFAR10(MyDummyDataset):
    def __init__(self, corruption, level, transform=None):
        if corruption == 'original':
            self.ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=None)
        else:
            self.ds = CIFAR10_C(corruption=corruption, level=level, transform=None)


class CIFAR100(MyDummyDataset):
    def __init__(self, corruption, level, transform=None):
        if corruption == 'original':
            self.ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=None)
        else:
            self.ds = CIFAR100_C(corruption=corruption, level=level, transform=None)


class CorruptionDataset(MyDummyDataset):
    def __init__(self, dataset, corruption, level) -> None:
        if dataset.lower() == 'cifar10':
            self.ds = CIFAR10(corruption, level)
        elif  dataset.lower() == 'cifar100':
            self.ds = CIFAR100(corruption, level)
        elif dataset.lower() == 'imagenet':
            self.ds = Imagenet(corruption, level)
