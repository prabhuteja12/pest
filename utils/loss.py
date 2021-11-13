import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(logits):
    return -1 * (logits.softmax(1) * logits.log_softmax(1)).sum() / logits.size(0)


class JSDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_aug_1, logits_aug_2, logits_clean=None):
        p_clean = logits_clean.softmax(1) if torch.is_tensor(logits_clean) else logits_clean
        p_aug1, p_aug2 = logits_aug_1.softmax(1), logits_aug_2.softmax(1)

        if p_clean is not None:
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()
        else:
            p_mixture = torch.clamp((p_aug1 + p_aug2) / 2.0, 1e-7, 1).log()

        loss = 0.0
        loss += F.kl_div(p_mixture, p_aug1, reduction="batchmean")
        loss += F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        loss += F.kl_div(p_mixture, p_clean, reduction="batchmean")

        return loss 


class EntropyJSD(JSDivergence):
    def __init__(self, loss_lambda=1.0):
        super().__init__()
        self.reg = loss_lambda

    def forward(self, logits_aug_1, logits_aug_2, logits_clean):
        jsd = super().forward(logits_aug_1, logits_aug_2, logits_clean=logits_clean)
        
        return jsd + (self.reg * entropy(logits_clean))


class Entropy(nn.Module):
    def forward(self, x):
        return entropy(x)



def get_loss_function(name, **params):
    if name == 'jsd':
        return JSDivergence()
    elif name == 'entropy':
        return Entropy()    
    elif name == 'entropyjsd':
        return EntropyJSD(**params)    
    else:
        print(f'Got {name}')
        raise ValueError()