import torch.optim as optim


def get_optimizer(name, model, optim_parameters, freeze_classifier=False):
    optim_class = getattr(optim, name)
    if hasattr(model, 'optim_parameters'):
        params = model.optim_parameters(freeze_classifier=freeze_classifier, **optim_parameters)
    else:
        params = model.parameters()
    return optim_class(params, **optim_parameters)


def lr_scheduler(optimizer, curr_iter, total_iter, gamma=10, power=0.75):
    """ Dont ask how this formula came about. Just trust that this is 
    MOST probably correct"""
    for param_group in optimizer.param_groups:
        decay = ((total_iter + gamma * (curr_iter - 1)) / (total_iter + gamma * curr_iter)) ** power
        param_group["lr"] = param_group["lr"] * decay

    return optimizer
