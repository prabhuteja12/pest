import torch.nn as nn
from torchvision import models

from .digit_network import init_weights

vgg_dict = {
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "vgg11bn": models.vgg11_bn,
    "vgg13bn": models.vgg13_bn,
    "vgg16bn": models.vgg16_bn,
    "vgg19bn": models.vgg19_bn,
}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50": models.resnext50_32x4d,
    "resnext101": models.resnext101_32x8d,
}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super().__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class FeatBottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, class_num=12):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class ObjectNetwork(nn.Module):
    def __init__(self, resname, bottleneck_dim=256, class_num=10):
        super().__init__()
        self.base = ResBase(resname)
        self.bn = FeatBottleneck(self.base.in_features, bottleneck_dim=bottleneck_dim, class_num=class_num)

    def forward(self, x):
        return self.bn(self.base(x))

    def optim_parameters(self, lr=1e-3, momentum=0.9, weight_decay=1e-3, freeze_classifier=False):
        param_group = []
        for v in self.base.parameters():
            param_group += [{"params": v, "lr": lr * 0.1, 'momentum': momentum, 'weight_decay': weight_decay}]

        if not freeze_classifier:
            for v in self.bn.parameters():
                param_group += [{"params": v, "lr": lr, 'momentum': momentum, 'weight_decay': weight_decay}]
        return param_group
