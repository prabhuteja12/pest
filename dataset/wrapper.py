import torch.utils.data as data
from torchvision import transforms 

from .cifar import CorruptionDataset, cifar_transform, imagenet_transform
from .visda import VisDaTest, visda_test_transforms
from .adversarial import ImagenetAdversarial, imageneta_transforms

from .randaugment import RandAugment
from .augmix import AugMix


class WrapperDataset(data.Dataset):
    def __init__(self, dataset, augmentations, transforms=None, multi_out=True):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.augmentations = augmentations if transforms else lambda *args: augmentations(args[0])
        self.multi_out = multi_out

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.multi_out:
            im_tuple = (self.transforms(x), self.augmentations(x), self.augmentations(x))
        else:
            im_tuple = (self.augmentations(x), )

        return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset, augmentation, corruption=None, level=None, **aug_args):
    if dataset == 'visda':
        dataset = VisDaTest()
        transform = visda_test_transforms

    elif dataset in ['imagenet', 'cifar100', 'cifar10']:
        transform = imagenet_transform if dataset == 'imagenet' else cifar_transform
        dataset = CorruptionDataset(dataset, corruption=corruption, level=level)

    elif dataset == 'imageneta':
        transform = imageneta_transforms
        dataset = ImagenetAdversarial()

    if augmentation.lower() == 'randaugment':
        augmentation = transforms.Compose([RandAugment(**aug_args), transform])
    elif augmentation.lower() == 'augmix':
        augmentation = AugMix(base_transforms=transform, **aug_args)

    return WrapperDataset(dataset, augmentations=augmentation, transforms=transform)

