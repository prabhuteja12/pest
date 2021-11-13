import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, model, **unused):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, x):
        clean = torch.split(x, int(x.size(0) / 3))[0] 
        return self.model(clean)
