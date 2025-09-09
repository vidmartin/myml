
from typing import *
from abc import ABC, abstractmethod
import functools
import itertools
import numpy as np

def preview_array(arr: np.ndarray):
    np.set_printoptions(precision=4, suppress=True)
    indexer = (slice(0, 5),) * len(arr.shape)
    return arr[indexer]

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

def ravel_dims(arr: np.ndarray, dims: Iterable[int]) -> np.ndarray:
    dims = tuple(dims)
    dims_set = set(dims)
    other_dims = tuple(a for a in range(len(arr.shape)) if a not in dims_set)
    arr_t = np.transpose(arr, other_dims + dims)
    return arr_t.reshape(arr_t.shape[:len(other_dims)] + (-1,))

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

def roll_varied(array: np.ndarray, roll_dim: int, vary_dim: int, offsets: Iterable[int]) -> np.ndarray:
    indexer = list(np.ix_(*[np.arange(v) for v in array.shape]))

    offsets_arr = np.array(list(offsets))
    offsets_desired_shape = np.ones(len(array.shape), dtype=np.int32)
    offsets_desired_shape[vary_dim] = offsets_arr.shape[0]
    offsets_arr = offsets_arr.reshape(tuple(offsets_desired_shape))

    # print(f"{indexer[roll_dim].shape} vs. {offsets_arr.shape}")
    indexer[roll_dim] = indexer[roll_dim] - offsets_arr
    indexer[roll_dim] %= array.shape[roll_dim]
    
    return array[tuple(indexer)]

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
# TODO: ^ delete?

def convolution_v2(array: np.ndarray, kernel: np.ndarray, stride: tuple[int, ...]):
    assert len(kernel.shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel.shape)

    view = np.lib.stride_tricks.sliding_window_view(
        array, kernel.shape, tuple(non_spatial_dims + i for i in range(len(kernel.shape)))
    )
    view_indexer = (slice(0, None),) * non_spatial_dims + tuple(slice(0, None, s) for s in stride) + (...,)
    view_ = view[view_indexer]
    return np.tensordot(view_, kernel, len(kernel.shape))
# TODO: ^ delete?

CONVOLUTION_FUNCTIONS_BY_VERSION = {
    1: convolution,
    2: convolution_v2,
}
# TODO: ^ delete?

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
# TODO: ^ delete?

def transposed_convolution_v2(array: np.ndarray, kernel: np.ndarray, stride: tuple[int, ...]):
    assert len(kernel.shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel.shape)

    out_shape = array.shape[:non_spatial_dims] + tuple(
        kernel.shape[meta_i] + stride[meta_i] * (array.shape[non_spatial_dims + meta_i] - 1)
        for meta_i in range(len(kernel.shape))
    )

    result = np.zeros(out_shape)
    result_view = np.lib.stride_tricks.sliding_window_view(
        result,
        kernel.shape,
        tuple(i + non_spatial_dims for i in range(len(kernel.shape))),
        writeable=True,
    )
    view_indexer = (slice(0, None),) * non_spatial_dims + tuple(slice(0, None, s) for s in stride) + (...,)
    result_view_ = result_view[view_indexer]

    array_indexer = (slice(0, None),) * len(array.shape) + (np.newaxis,) * len(kernel.shape)
    np.add.at(result_view_, (slice(0, None),) * len(result_view_.shape), array[array_indexer] * kernel)
    return result
# TODO: ^ delete?

def transposed_convolution_v3(array: np.ndarray, kernel: np.ndarray, stride: tuple[int, ...]):
    assert len(kernel.shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel.shape)

    # if stride == (1,) * len(stride):
    #     array_ = np.zeros(
    #         array.shape[:non_spatial_dims] + tuple(
    #             2 * (kernel.shape[meta_i] - 1) + array.shape[non_spatial_dims + meta_i]
    #             for meta_i in range(len(kernel.shape))
    #         )
    #     )
    #     indexer = (slice(0, None),) * non_spatial_dims + tuple(
    #         slice(kernel.shape[meta_i] - 1, kernel.shape[meta_i] - 1 + array.shape[non_spatial_dims + meta_i])
    #         for meta_i in range(len(kernel.shape))
    #     )
    #     array_[indexer] = array
    #     return convolution_v2(array_, np.flip(kernel), stride)

    out_shape = array.shape[:non_spatial_dims] + tuple(
        kernel.shape[meta_i] + stride[meta_i] * (array.shape[non_spatial_dims + meta_i] - 1)
        for meta_i in range(len(kernel.shape))
    )

    ker_ = np.zeros(tuple(
        (kernel.shape[meta_i] + stride[meta_i] - 1) // stride[meta_i] * stride[meta_i]
        for meta_i in range(len(kernel.shape))
    ))
    indexer = tuple(slice(0, kernel.shape[meta_i]) for meta_i in range(len(kernel.shape)))
    ker_[indexer] = kernel

    mini_kernel_shape = tuple(
        ker_.shape[meta_i] // stride[meta_i]
        for meta_i in range(len(kernel.shape))
    )
    multi_kernel = np.zeros(mini_kernel_shape + stride)
    for idx in itertools.product(*[range(s) for s in stride]):
        indexer = tuple(slice(idx[meta_i], None, stride[meta_i]) for meta_i in range(len(kernel.shape)))
        multi_kernel[(slice(0, None),) * len(mini_kernel_shape) + idx] = np.flip(ker_[indexer])

    array_ = np.zeros(
        array.shape[:non_spatial_dims] + tuple(
            2 * (mini_kernel_shape[meta_i] - 1) + array.shape[non_spatial_dims + meta_i]
            for meta_i in range(len(kernel.shape))
        )
    )
    indexer = (slice(0, None),) * non_spatial_dims + tuple(
        slice(mini_kernel_shape[meta_i] - 1, mini_kernel_shape[meta_i] - 1 + array.shape[non_spatial_dims + meta_i])
        for meta_i in range(len(kernel.shape))
    )
    array_[indexer] = array

    view = np.lib.stride_tricks.sliding_window_view(
        array_, mini_kernel_shape,
        tuple(non_spatial_dims + meta_i for meta_i in range(len(kernel.shape)))
    )
    res: np.ndarray = np.tensordot(view, multi_kernel, len(kernel.shape))

    res = res.transpose(
        tuple(range(non_spatial_dims)) + tuple(
            non_spatial_dims + meta_i + d * len(kernel.shape)
            for meta_i in range(len(kernel.shape))
            for d in (0, 1)
        )
    ) # transpose so that spatial dimension index and corresponding "minikernel index" are next to each other
    # so that reshape does the right thing

    res = res.reshape(
        array.shape[:non_spatial_dims] + tuple(
            stride[meta_i] * (array.shape[non_spatial_dims + meta_i] + mini_kernel_shape[meta_i] - 1)
            for meta_i in range(len(kernel.shape))
        )
    )

    indexer = tuple(slice(0, out_shape[meta_i]) for meta_i in range(len(array.shape)))
    return res[indexer]
# TODO: ^ delete?

TRANSPOSED_CONVOLUTION_FUNCTIONS_BY_VERSION = {
    1: transposed_convolution,
    2: transposed_convolution_v2,
}
# TODO: ^ delete?

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

def multichannel_convolution_v2(array: np.ndarray, kernels: np.ndarray, stride: tuple[int, ...]):
    out_channels, in_channels, *kernel_shape = kernels.shape
    kernel_shape = tuple(kernel_shape)
    assert len(kernel_shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel_shape) - 1 # the -1 is the channels dimension

    view = np.lib.stride_tricks.sliding_window_view(
        array,
        kernel_shape,
        tuple(i + non_spatial_dims + 1 for i in range(len(kernel_shape))),
    )
    view = np.moveaxis(view, -2 * len(kernel_shape) - 1, -len(kernel_shape) - 1) # move channels dim between "output pos dims" and "kernel dims"
    kernels_ = np.moveaxis(kernels, 0, -1) # move out channels to back
    view_indexer = (slice(0, None),) * non_spatial_dims + tuple(slice(0, None, s) for s in stride) + (...,)
    result = np.tensordot(view[view_indexer], kernels_, len(kernel_shape) + 1) # contract over the kernel dims as well as input channel dims
    result = np.moveaxis(result, -1, -1 - len(kernel_shape)) # move output channels dim to where it's supposed to be
    return result

MULTICHANNEL_CONVOLUTION_FUNCTIONS_BY_VERSION: Dict[int, Callable[[np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray]] = {
    1: multichannel_convolution,
    2: multichannel_convolution_v2,
}

def multichannel_transposed_convolution(array: np.ndarray, kernels: np.ndarray, stride: tuple[int, ...]):
    out_channels, in_channels, *kernel_shape = kernels.shape
    kernel_shape = tuple(kernel_shape)
    assert len(kernel_shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel_shape) - 1
    
    out_shape = array.shape[:non_spatial_dims] + (in_channels,) + tuple(
        kernel_shape[meta_i] + stride[meta_i] * (array.shape[non_spatial_dims + meta_i + 1] - 1)
        for meta_i in range(len(kernel_shape))
    )

    result = np.zeros(out_shape)
    for idx in itertools.product(*[range(k) for k in kernel_shape]):
        result_indexer = (slice(0, None),) * (non_spatial_dims + 1) + tuple(
            slice(
                idx[meta_i],
                idx[meta_i] + stride[meta_i] * array.shape[non_spatial_dims + meta_i + 1],
                stride[meta_i]
            )
            for meta_i in range(len(idx))
        )

        arr = np.tensordot(array, kernels[:,:,*idx], ((non_spatial_dims,), (0,)))
        arr = np.moveaxis(arr, -1, non_spatial_dims)
        result[result_indexer] += arr
    return result

def multichannel_transposed_convolution_v2(array: np.ndarray, kernel: np.ndarray, stride: tuple[int, ...]):
    out_channels, in_channels, *kernel_shape = kernel.shape
    kernel_shape = tuple(kernel_shape)
    assert len(kernel_shape) == len(stride)
    non_spatial_dims = len(array.shape) - len(kernel_shape) - 1

    out_shape = array.shape[:non_spatial_dims] + (in_channels,) + tuple(
        kernel_shape[meta_i] + stride[meta_i] * (array.shape[non_spatial_dims + meta_i + 1] - 1)
        for meta_i in range(len(kernel_shape))
    )

    kernel_adjusted_shape = tuple(
        (kernel_shape[meta_i] + stride[meta_i] - 1) // stride[meta_i] * stride[meta_i]
        for meta_i in range(len(kernel_shape))
    )
    kernel_adjusted = np.zeros((out_channels, in_channels) + kernel_adjusted_shape)
    indexer = (slice(0, None),) * 2 + tuple(
        slice(0, kernel_shape[meta_i])
        for meta_i in range(len(kernel_shape))
    )
    kernel_adjusted[indexer] = kernel

    mini_kernel_shape = tuple(
        kernel_adjusted_shape[meta_i] // stride[meta_i]
        for meta_i in range(len(kernel_shape))
    )
    multi_kernel = np.zeros((out_channels, in_channels) + mini_kernel_shape + stride)
    for idx in itertools.product(*[range(s) for s in stride]):
        dest_indexer = (slice(0, None), slice(0, None)) + (slice(0, None),) * len(kernel_shape) + idx
        src_indexer = (slice(0, None), slice(0, None)) + tuple(
            slice(idx[meta_i], None, stride[meta_i])
            for meta_i in range(len(kernel_shape))
        )
        multi_kernel[dest_indexer] = np.flip(kernel_adjusted[src_indexer], tuple(range(2, len(kernel.shape))))

    array_ = np.zeros(
        array.shape[:non_spatial_dims] + (out_channels,) + tuple(
            2 * (mini_kernel_shape[meta_i] - 1) + array.shape[non_spatial_dims + meta_i + 1]
            for meta_i in range(len(kernel_shape))
        )
    )
    indexer = (slice(0, None),) * non_spatial_dims + (slice(0, None),) + tuple(
        slice(mini_kernel_shape[meta_i] - 1, mini_kernel_shape[meta_i] - 1 + array.shape[non_spatial_dims + meta_i + 1])
        for meta_i in range(len(kernel_shape))
    )
    array_[indexer] = array

    view: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        array_, mini_kernel_shape,
        tuple(non_spatial_dims + 1 + meta_i for meta_i in range(len(kernel_shape)))
    )
    view = np.moveaxis(view, -(2 * len(kernel_shape)) - 1, -len(kernel_shape) - 1)
    multi_kernel_t = multi_kernel.transpose((0,) + tuple(range(2, len(multi_kernel.shape))) + (1,))
    print(f"{view.shape} * {multi_kernel_t.shape}")
    res: np.ndarray = np.tensordot(view, multi_kernel_t, len(kernel_shape) + 1)
    print(res.shape)
    res = res.transpose(
        tuple(range(non_spatial_dims)) + (len(res.shape) - 1,) + tuple(
            non_spatial_dims + meta_i + d * len(kernel_shape)
            for meta_i in range(len(kernel_shape))
            for d in (0, 1)
        )
    ) # transpose so that spatial dimension index and corresponding "minikernel index" are next to each other
    # so that reshape does the right thing
    print(res.shape)

    res = res.reshape(
        array.shape[:non_spatial_dims] + (in_channels,) + tuple(
            stride[meta_i] * (array.shape[non_spatial_dims + 1 + meta_i] + mini_kernel_shape[meta_i] - 1)
            for meta_i in range(len(kernel_shape))
        )
    )
    print(res.shape)

    indexer = tuple(slice(0, out_shape[meta_i]) for meta_i in range(len(array.shape)))
    return res[indexer]

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
