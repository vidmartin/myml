
from typing import *
from abc import ABC, abstractmethod
import functools
import itertools
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

def pad_l(array: np.ndarray, padding: tuple[int, ...], fill_value: float):
    non_spatial_dims = len(array.shape) - len(padding)
    padded_array = np.full(
        array.shape[:non_spatial_dims] + tuple(
            v + p for p, v in zip(
                padding, array.shape[non_spatial_dims:]
            )
        ),
        fill_value
    )
    padded_array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(p, None) for p in padding
    )
    padded_array[padded_array_indexer] = array
    return padded_array

def pad_r(array: np.ndarray, padding: tuple[int, ...], fill_value: float):
    non_spatial_dims = len(array.shape) - len(padding)
    padded_array = np.full(
        array.shape[:non_spatial_dims] + tuple(
            v + p for p, v in zip(
                padding, array.shape[non_spatial_dims:]
            )
        ),
        fill_value
    )
    padded_array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(0, -p) if p != 0 else slice(0, None)
        for p in padding
    )
    padded_array[padded_array_indexer] = array
    return padded_array

def pad_lr(array: np.ndarray, padding: tuple[int, ...], fill_value: float):
    non_spatial_dims = len(array.shape) - len(padding)
    padded_array = np.full(
        array.shape[:non_spatial_dims] + tuple(
            v + 2 * p for p, v in zip(
                padding, array.shape[non_spatial_dims:]
            )
        ),
        fill_value
    )
    padded_array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(p, -p) if p != 0 else slice(0, None)
        for p in padding
    )
    padded_array[padded_array_indexer] = array
    return padded_array

def unpad_l(padded_array: np.ndarray, padding: tuple[int, ...]):
    non_spatial_dims = len(padded_array.shape) - len(padding)
    padded_array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(p, None) for p in padding
    )
    return padded_array[padded_array_indexer]

def unpad_r(padded_array: np.ndarray, padding: tuple[int, ...]):
    non_spatial_dims = len(padded_array.shape) - len(padding)
    padded_array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(0, -p) if p != 0 else slice(0, None)
        for p in padding
    )
    return padded_array[padded_array_indexer]

def unpad_lr(padded_array: np.ndarray, padding: tuple[int, ...]):
    non_spatial_dims = len(padded_array.shape) - len(padding)
    padded_array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(p, -p) if p != 0 else slice(0, None)
        for p in padding
    )
    return padded_array[padded_array_indexer]

def convolution(array: np.ndarray, kernel: np.ndarray, stride: tuple[int, ...]):
    assert len(kernel.shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel.shape)

    # TODO: assert kernel small enough
    out_shape = array.shape[:non_spatial_dims] + tuple(
        1 + (array.shape[non_spatial_dims + meta_i] - kernel.shape[meta_i]) // stride[meta_i]
        for meta_i in range(len(kernel.shape))
    )

    result = np.zeros(out_shape)
    for idx in itertools.product(*[range(k) for k in kernel.shape]):
        array_indexer = (slice(0, None),) * non_spatial_dims + tuple(
            slice(
                idx[meta_i],
                idx[meta_i] + stride[meta_i] * out_shape[non_spatial_dims + meta_i],
                stride[meta_i]
            )
            for meta_i in range(len(idx))
        )
        result += kernel[idx] * array[array_indexer]
    return result

def transposed_convolution(array: np.ndarray, kernel: np.ndarray, stride: tuple[int, ...]):
    assert len(kernel.shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel.shape)
    
    out_shape = array.shape[:non_spatial_dims] + tuple(
        kernel.shape[meta_i] + stride[meta_i] * (array.shape[non_spatial_dims + meta_i] - 1)
        for meta_i in range(len(kernel.shape))
    )

    result = np.zeros(out_shape)
    for idx in itertools.product(*[range(k) for k in kernel.shape]):
        result_indexer = (slice(0, None),) * non_spatial_dims + tuple(
            slice(
                idx[meta_i],
                idx[meta_i] + stride[meta_i] * array.shape[non_spatial_dims + meta_i],
                stride[meta_i]
            )
            for meta_i in range(len(idx))
        )

        result[result_indexer] += kernel[idx] * array
    return result

def multichannel_convolution(array: np.ndarray, kernels: np.ndarray, stride: tuple[int, ...]):
    out_channels, in_channels, *kernel_shape = kernels.shape
    kernel_shape = tuple(kernel_shape)
    assert len(kernel_shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel_shape) - 1 # the -1 is the channels dimension

    # TODO: assert kernel small enough
    out_shape = array.shape[:non_spatial_dims] + (out_channels,) + tuple(
        1 + (array.shape[non_spatial_dims + meta_i + 1] - kernel_shape[meta_i]) // stride[meta_i]
        for meta_i in range(len(kernel_shape))
    )

    result = np.zeros(out_shape)
    for idx in itertools.product(*[range(k) for k in kernel_shape]):
        array_indexer = (slice(0, None),) * (non_spatial_dims + 1) + tuple(
            slice(
                idx[meta_i],
                idx[meta_i] + stride[meta_i] * out_shape[non_spatial_dims + 1 + meta_i],
                stride[meta_i]
            )
            for meta_i in range(len(idx))
        )
        arr = np.tensordot(array[array_indexer], kernels[:,:,*idx], ((non_spatial_dims,), (1,)))
        arr = np.moveaxis(arr, -1, non_spatial_dims)
        result += arr
    return result

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
