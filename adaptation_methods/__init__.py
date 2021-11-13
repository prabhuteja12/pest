from .baseline import Baseline
from .noise_robustness import NoiseRobustAdaptation
from .norm import NormTraining
from .tent import TentTraining


_methods = {}
_methods['baseline'] = Baseline
_methods['tent'] = TentTraining
_methods['norm'] = NormTraining
_methods['noising'] = NoiseRobustAdaptation


def get_adaptation_method(method):
    return _methods[method]