import robustbench


dataset_specifics = {}

dataset_specifics["cifar10"] = {
    # "corruption": robustbench.data.CORRUPTIONS,
    "corruption": ["impulse_noise", "gaussian_noise"],
    "level": list(range(1, 6))[-1:],
    "dataset": ["cifar10"],
    "train_batch_size": [200],
}
dataset_specifics["cifar100"] = {
    "corruption": robustbench.data.CIFAR_100_CORRUPTIONS,
    "level": list(range(1, 6))[-1:],
    "dataset": ["cifar100"],
    "train_batch_size": [200],
}
dataset_specifics["imagenet"] = {
    "corruption": robustbench.data.CIFAR_100_CORRUPTIONS,
    "level": list(range(1, 6)),
    "dataset": ["imagenet"],
    "train_batch_size": [200],
}
dataset_specifics["visda"] = {"dataset": ["visda"]}

dataset_specifics["imageneta"] = {}
