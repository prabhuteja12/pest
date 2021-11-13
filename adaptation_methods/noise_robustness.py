import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    freeze_except_bn,
    get_optimizer,
    get_loss_function,
    copy_model_and_optimizer,
    load_model_and_optimizer,
)


class NoiseRobustAdaptation(nn.Module):
    def __init__(
        self,
        model,
        episodic,
        num_steps,
        freeze_bn,
        freeze_classifier,
        loss_fn,
        optimizer,
        optim_parameters,
        **unused
    ):
        super().__init__()
        self.model = model
        self.episodic = episodic
        if freeze_bn:
            self.model = freeze_except_bn(self.model)
        self.optimizer = get_optimizer(
            name=optimizer,
            model=self.model,
            optim_parameters=optim_parameters,
            freeze_classifier=freeze_classifier,
        )
        self.crit = get_loss_function(loss_fn)
        self.num_steps = num_steps
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.num_steps):
            _ = forward_and_adapt(x, self.crit, self.model, self.optimizer)
        self.model.eval()
        with torch.no_grad():
            ret = self.model(torch.split(x, int(x.size(0) / 3))[0])
        self.model.train()
        return ret

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


@torch.enable_grad()
def forward_and_adapt(x, criterion, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    logits_clean, logits_aug1, logits_aug2 = torch.split(outputs, x.size(0) // 3)
    # adapt
    loss = criterion(logits_aug1, logits_aug2, logits_clean)
    loss.backward()
    optimizer.step()
    model.eval()
    return outputs
