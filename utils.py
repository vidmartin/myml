
from typing import *
import functools
from abc import ABC, abstractmethod
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

def normalize_dest_shape(src_shape: tuple[int, ...], dest_shape: tuple[int, ...]):
    n_jokers = sum(1 for v in dest_shape if v < 0)
    src_volume = functools.reduce(lambda x, y: x * y, src_shape)
    dest_part_volume = functools.reduce(lambda x, y: x * y, [v for v in dest_shape if v >= 0])
    assert (n_jokers == 1 and src_volume % dest_part_volume == 0) or (n_jokers == 0 and src_volume == dest_part_volume)
    return tuple(
        v if v >= 0 else src_volume // dest_part_volume
        for v in dest_shape
    )

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

class RandomGenerator(ABC):
    @abstractmethod
    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        """Returns a random numpy array of the given shape."""
        raise NotImplementedError()
    
class NumpyUniformRandomGenerator(RandomGenerator):
    def __init__(self, rng: np.random.Generator | None):
        self._rng = rng
    @override
    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._rng is not None:
            return self._rng.random(shape)
        return np.random.random(shape)

try:
    # TODO: check if this is how optional dependencies are properly handled

    import torch

    class TorchUniformRandomGenerator(RandomGenerator):
        @override
        def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
            rand_tensor = torch.rand(shape, requires_grad=False)
            return rand_tensor.numpy()
except ImportError:
    class TorchUniformRandomGenerator(RandomGenerator):
        @override
        def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
            raise NotImplementedError("this class can't be used when PyTorch is not imported")
