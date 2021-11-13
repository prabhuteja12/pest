import torch
import torch.nn as nn

from utils import freeze_except_bn, get_optimizer, get_loss_function


class TentTraining(nn.Module):
    def __init__(self, model, optimizer, num_steps, optim_parameters, freeze_classifier, **unused):
        super().__init__()
        self.model = freeze_except_bn(model)
        self.optimizer = get_optimizer(
            name=optimizer, model=self.model, optim_parameters=optim_parameters, freeze_classifier=freeze_classifier
        )
        self.steps = num_steps
        self.crit = get_loss_function("entropy")

    def forward(self, x):
        x = torch.split(x, int(x.size(0) / 3))[0]
        for _ in range(self.steps):
            _ = forward_and_adapt(x, self.crit, self.model, self.optimizer)
        return self.model(x)


@torch.enable_grad()
def forward_and_adapt(x, criterion, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = criterion(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs
