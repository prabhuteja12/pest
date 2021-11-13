import itertools
from collections import ChainMap

import pandas as pd


def unformat_json(json, sep="."):
    parsed = {}
    for label, v in json.items():
        keys = label.split(".")
        current = parsed
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current[k] = v
            else:
                if k not in current.keys():
                    current[k] = {}
                current = current[k]
    return parsed


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def generate_configurations(template, *possibilities):
    template = dict(template)
    flattened = pd.json_normalize(template, sep=".").to_dict(orient="records")[0]
    possibilities = ChainMap(*possibilities)

    cleaned_possibilites = {}
    for k, v in possibilities.items():
        for full_key in flattened.keys():
            if k in full_key.split('.'):
                cleaned_possibilites[full_key] = v
    all_perms = []
    for new_vals in dict_product(cleaned_possibilites):
        flattened.update(new_vals)
        all_perms.append(unformat_json(flattened.copy()))

    return all_perms
