import collections.abc
import os
from pprint import pprint

import yaml
from tunables import defaults


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping) and k != 'augmentation_params':
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_config(filename):
    with open(filename, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    default = defaults[cfg["dataset"]]
    # default.update(cfg)
    default = update(default, cfg)
    return dotdict(default)


def save_config(config, filename):
    with open(filename, "w") as fp:
        yaml.dump(config, fp)


def save_multi_config(configs, folder="./cfs"):
    currfiles = [x for x in os.listdir(folder) if x.endswith("yaml")]
    currfiles.sort(key=lambda fname: int(fname.split(".")[0]))

    last = int(currfiles[-1].split(".")[0]) if len(currfiles) > 0 else 0
    save_index = last + 1

    for x in configs:
        save_config(x, f"{folder}/{save_index}.yaml")
        save_index += 1

    return last + 1, save_index - 1
