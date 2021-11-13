import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

# ImageNet code should change this value
IMAGE_SIZE = 32


def to_numpy(pil_img):
    return np.array(pil_img)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
        An int that results from scaling `maxval` according to `level`.
      """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y]

augmentations_all = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]


def apply_op(image, op, severity):
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.0


class AugMix:
    def __init__(self, base_transforms, severity=3, width=3, depth=-1, alpha=1.0) -> None:
        self.base_transforms = base_transforms
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
    
    def __call__(self, img):
        return augment_and_mix(img, self.base_transforms, self.severity, self.width, self.depth, self.alpha)


def augment_and_mix(image, preproc, severity=3, width=3, depth=-1, alpha=1.0):
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    mix = 0
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else random.randint(1, 4)

        for _ in range(d):
            op = random.choice(augmentations)
            image_aug = op(image_aug, severity)

        mix += ws[i] * preproc(image_aug)
    mixed = (1 - m) * preproc(image) + m * mix

    return mixed


# class AugMixDataset(torch.utils.data.Dataset):
#     """Dataset wrapper to perform AugMix augmentation."""

#     def __init__(self, dataset, corruption, level, preprocess=te_transforms, no_jsd=False):
#         if dataset == "cifar10":
#             tesize = 10000
#             if corruption in (None, "original"):
#                 print("Test on the original test set")
#                 teset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
#             elif corruption in common_corruptions:
#                 print("Test on %s level %d" % (corruption, level))
#                 teset = CIFAR10_C(corruption=corruption, level=level)
                
#             elif corruption == "cifar_new":
#                 print("Test on CIFAR-10.1")
#                 teset = CIFAR10_1()
#             else:
#                 raise Exception("Corruption not found!")
#         else:
#             raise Exception("Dataset not found!")
#         self.dataset = teset
#         self.preprocess = preprocess
#         self.no_jsd = no_jsd

#     def __getitem__(self, i):
#         x, y = self.dataset[i]
#         if self.no_jsd:
#             return augment_and_mix(x, self.preprocess), y
#         else:
#             im_tuple = (self.preprocess(x), augment_and_mix(x, self.preprocess), augment_and_mix(x, self.preprocess))
#             return im_tuple, y

#     def __len__(self):
#         return len(self.dataset)