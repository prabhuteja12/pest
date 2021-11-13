import os

import torch
from robustbench.utils import load_model
from torchvision.models import resnet50, densenet121

from .object_network import ObjectNetwork

CIFAR10_MODELS = {
    "resnet20": lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True),
    "wrn28": lambda: load_model("Standard", "downloaded_nets/torch", "cifar10", "corruptions"),
    "wrn40": lambda: load_model("Hendrycks2020AugMix_WRN", "downloaded_nets/torch", "cifar10", "corruptions")
}

CIFAR100_MODELS = {
    "wrn40": lambda: load_model("Hendrycks2020AugMix_WRN", "downloaded_nets/torch", "cifar100", "corruptions")
}

IMAGENET_MODELS = {
    'resnet50': lambda: resnet50(pretrained=True),
    'densenet121': lambda: densenet121(pretrained=True)
}



def get_model(dataset, model_name=None, **kwargs):
    if dataset == "cifar10":
        return CIFAR10_MODELS[model_name or 'wrn28']()

    if dataset == "cifar100":
        return CIFAR100_MODELS[model_name or 'wrn40']()

    if dataset in ['imagenet', 'imageneta']:
        return IMAGENET_MODELS[model_name or 'resnet50']()
    
    if dataset == 'visda':
        model = ObjectNetwork('resnet50', class_num=12)
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pretrained', 'visda_resnet50.pth')
        sd = torch.load(load_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(sd)
        return model