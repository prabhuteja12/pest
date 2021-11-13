aug_methods = {}

aug_methods["augmix"] = {
    "augmentation": ["augmix"],
    "alpha": [1],
    "depth": [1, 3][1:],
    "severity": [1, 2, 3][:1],  # [2:],
    "width": [1, 2, 3][2:],  # [1:2],
}

aug_methods["randaugment"] = {
    "augmentation": ["randaugment"],
    "m": [1, 3][:1],
    "n": [1, 3][:1],
}
