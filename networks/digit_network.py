import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class FeatBottleneck(nn.Module):
    def __init__(self, feature_dim, class_num, bottleneck_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True) if type == "bn" else nn.Identity()
        self.dropout = nn.Dropout(p=0.5) if type == "bn" else nn.Identity()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x)


class Classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super().__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
        )
        self.in_features = 256 * 4 * 4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x



class DigitNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()
        self.bottle = FeatBottleneck(self.features.in_features, 10)
        # self.clf = Classifier(10)

    def forward(self, x):
        x = self.features(x)
        x = self.bottle(x)
        return x