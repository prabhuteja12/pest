import math

import torch
from tqdm import tqdm


def accuracy_samples(
    model, x, y, batch_size, device=None
):
    if device is None:
        device = x.device
    acc = 0.0
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size : (counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size : (counter + 1) * batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def accuracy_clean(model, dataset, batch_size, class_subset=None, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    acc = 0.0
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=8)
    # model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            if isinstance(x, (list, tuple)):
                x = torch.cat(x, 0).to(device)
            else:
                x = x.to(device)
            pred = model(x)

            ### Two lines to handle ImageNet Adversarial. See hendrycks eval.py at 
            ### https://github.com/hendrycks/natural-adv-examples/blob/master/eval.py
            if class_subset is not None: 
                pred = pred[:, class_subset]
            pred = pred.argmax(1).cpu()
            acc += (pred == y.cpu()).float().sum()
    
    return acc.item() / len(dataset)

