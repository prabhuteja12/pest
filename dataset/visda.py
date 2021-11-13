import os.path as osp
from functools import partial
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from .tar_dataset import TarImageFolder, TarDataset


class TarImageFolderList(TarDataset):
    def __init__(self, archive, imagelist_file, transform=None, is_valid_file=None, root_in_archive=""):
        if root_in_archive and not root_in_archive.endswith("/"):
            root_in_archive = root_in_archive + "/"
        self.root_in_archive = root_in_archive

        # load the archive meta information, and filter the samples
        super().__init__(archive=archive, transform=transform, is_valid_file=is_valid_file)
        self.class_to_idx = {}
        self.targets = []
        with open(imagelist_file, "r") as fp:
            all_images = fp.readlines()
        all_images = [x.strip().split() for x in all_images]
        self.samples = tuple(["test/" + x[0] for x in all_images])
        self.transform = transform
        self.targets = [s[1] for s in all_images]

    def filter_samples(self, is_valid_file=None, extensions=(".png", ".jpg", ".jpeg")):
        super().filter_samples(is_valid_file, extensions)
        self.samples = [filename for filename in self.samples if filename.startswith(self.root_in_archive)]

    def __getitem__(self, index):
        image = self.get_image(self.samples[index], pil=True)
        image = image.convert("RGB")  # if it's grayscale, convert to RGB
        if self.transform:  # apply any custom transforms
            image = self.transform(image)

        label = self.targets[index]

        return image, int(label)

resize_size=256 
crop_size=224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
visda_train_transforms = transforms.Compose(
    [
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

visda_test_transforms = transforms.Compose(
    [
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]
)


VisdaTrain = partial(TarImageFolder, "datasets/visda/train.tar", root_in_archive="train/")
VisdaValidation = partial(TarImageFolder, "datasets/visda/validation.tar", root_in_archive="validation/")
VisDaTest = partial(TarImageFolderList, "datasets/visda/test.tar", "datasets/visda/image_list.txt")

