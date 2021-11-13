methods = {}
methods["baseline"] = {"adapt_method": ["baseline"]}
methods["norm"] = {"adapt_method": ["norm"]}
methods["tent"] = {
    "num_steps": [1, 2, 3, 4, 5],
    "optimizer": ["Adam"],
    "lr": [10 ** -3],
    # "optim_params": [],
    "adapt_method": ["tent"],
    "loss_fn": ["entropy"]
}
methods["noising"] = {
    "num_steps": [1, 2, 3, 4, 5][4:],
    "optimizer": ["Adam", "SGD"][1:],
    # "lr": [10 ** -4, 5 * 10 ** -4, 10 ** -3, 10 ** -2][:1],
    # "lr": [10 **-6, 10 **-5, 10 **-4, 5 * 10 **-4, 10 **-3, 10 **-2, 10 **-1],
    "lr": [10 ** -4], 
    "freeze_classifier": [True, False][:1],
    "freeze_bn": [True, False][1:],
    "loss_fn": ["entropy", "jsd", "entropyjsd"][1:2],
    "loss_lambda": [10],
    "adapt_method": ["noising"],
    "episodic": [True, False][1:],
    "train_batch_size": [64]
}
methods["barlow"] = {
    "num_steps": [1, 2, 3, 4, 5],
    "optimizer": ["Adam", "SGD"],
    "lr": [10 ** -4, 5 * 10 ** -4, 10 ** -3, 10 ** -2],
    "freeze_classifier": [True, False],
    "freeze_bn": [True, False],
    "reg": [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2],
    "adapt_method": ["barlow"],
}

