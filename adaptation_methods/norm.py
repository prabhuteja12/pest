import torch
import torch.nn as nn

from utils import freeze_except_bn

class NormTraining(nn.Module):
    def __init__(self, model, **unused):
        super().__init__()
        self.model = model
        self.model = freeze_except_bn(self.model)

    def forward(self, x):
        clean = torch.split(x, int(x.size(0) / 3))[0]
        return self.model(clean)