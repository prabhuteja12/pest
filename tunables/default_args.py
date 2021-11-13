defaults = {}
defaults["cifar10"] = {
    "optimizer": "SGD",
    "optim_parameters": {"lr": 0.0004, "momentum": 0.9, "weight_decay": 0.0005},
    "augmentation": "randaugment",
    "augmentation_params": {"m": 1, "n": 1},
    "loss_fn": "entropyjsd",
    "loss_lambda": 100,
    "episodic": False,
    "train_batch_size": 200,
    "model": "wrn28",
}

defaults["cifar100"] = {
    "optimizer": "SGD",
    "optim_parameters": {"lr": 0.0004, "momentum": 0.9, "weight_decay": 0.0005},
    "augmentation": "randaugment",
    "augmentation_params": {"m": 1, "n": 1},
    "loss_fn": "entropyjsd",
    "loss_lambda": 100,
    "episodic": False,
    "train_batch_size": 200,
    "model": "wrn40",
}

defaults["imagenet"] = {
    "optimizer": "SGD",
    "optim_parameters": {"lr": 0.0004, "momentum": 0.9, "weight_decay": 0.0005},
    "augmentation": "randaugment",
    "augmentation_params": {"m": 3, "n": 3},
    "loss_fn": "entropyjsd",
    "loss_lambda": 100,
    "episodic": False,
    "train_batch_size": 200,
    "model": "resnet50"
}


defaults["imageneta"] = {
    "optimizer": "SGD",
    "optim_parameters": {"lr": 0.0004, "momentum": 0.9, "weight_decay": 0.0005},
    "augmentation": "randaugment",
    "augmentation_params": {"m": 3, "n": 3},
    "loss_fn": "entropyjsd",
    "loss_lambda": 100,
    "episodic": False,
    "train_batch_size": 200,
    "model": "resnet50"
}

defaults['visda'] = {
    "optimizer": "SGD",
    "optim_parameters": {"lr": 0.0004, "momentum": 0.9, "weight_decay": 0.0005},
    "augmentation": "augmix",
    "augmentation_params": {"m": 3, "n": 3},
    "loss_fn": "entropyjsd",
    "loss_lambda": 100,
    "episodic": True,
    "train_batch_size": 64,
    "model": "resnet50",
    "freeze_classifier": True
}

