
from typing import *
import numpy as np

def one_hot_encode(arr: np.ndarray, n_classes: int):
    ret = np.zeros(arr.shape + (n_classes,), dtype=np.float32)

    indexer = ()
    indices = np.indices(arr.shape)
    indexer = tuple(indices[i,...] for i in range(len(arr.shape))) + (arr,)
    ret[indexer] = 1.0

    return ret

def log_sum_exp(arr: np.ndarray):
    """Perform LogSumExp along the last dimension."""
    maxima = arr.max(-1)
    exps: np.ndarray = np.exp(arr - maxima[...,np.newaxis])
    return maxima + np.log(exps.sum(-1))

def softmax(arr: np.ndarray):
    """Perform softmax along the last dimension."""
    return np.exp(arr - log_sum_exp(arr)[...,np.newaxis])

T = TypeVar("T")
if False:
    def instance_memo(callable_: T) -> T:
        return callable_
else:
    def instance_memo(callable_):
        fn_name = callable_.__name__
        def inner(*args, **kwargs):
            assert not kwargs, "instance_memo doesn't support kwargs"
            assert args, "self argument is missing"
            self = args[0]

            attr_name = f"__memo_{fn_name}__"
            if not hasattr(self, attr_name):
                setattr(self, attr_name, {})
            attr_val = getattr(self, attr_name)
            assert isinstance(attr_val, dict)

            if args[1:] not in attr_val:
                attr_val[args[1:]] = callable_(*args)
            return attr_val[args[1:]]
        return inner
