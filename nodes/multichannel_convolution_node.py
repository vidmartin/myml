
from __future__ import annotations
from typing import *
import time
import itertools
import numpy as np
from nodes.lazy_dependent_node import LazyDependentNode
from nodes.tensor_node import TensorNode
from node_visitor.node_visitor import NodeVisitor
import utils

TResult = TypeVar("TResult")

# TODO: dilation

class MultichannelConvolutionNode(LazyDependentNode):
    def __init__(
        self,
        input_node: TensorNode,
        kernels_node: TensorNode,
        padding: tuple[int, ...],
        stride: tuple[int, ...],
        multichannel_convolution_function_or_version: Callable[[np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray] | int = 1,
        multichannel_transposed_convolution_function_or_version: Callable[[np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray] | int = 1
    ):
        super().__init__([input_node, kernels_node])
        self._input_node = input_node
        self._kernels_node = kernels_node
        self._out_channels, self._in_channels, *self._kernel_size = self._kernels_node.get_shape()
        self._kernel_size = tuple(self._kernel_size)
        self._padding = padding
        self._stride = stride
        self._input_shape, self._non_spatial_dims = self._init_and_check_shapes()
        self._padded_input_shape: tuple[int, ...] | None = None
        self._depval_padded: np.ndarray | None = None
        self._multichannel_convolution_function = \
            multichannel_convolution_function_or_version if not isinstance(multichannel_convolution_function_or_version, int) \
            else utils.MULTICHANNEL_CONVOLUTION_FUNCTIONS_BY_VERSION[multichannel_convolution_function_or_version]
        self._multichannel_transposed_convolution_functoion = \
            multichannel_transposed_convolution_function_or_version if not isinstance(multichannel_transposed_convolution_function_or_version, int) \
            else utils.MULTICHANNEL_TRANSPOSED_CONVOLUTION_FUNCTIONS_BY_VERSION[multichannel_transposed_convolution_function_or_version]
        self._time_forward: float | None = None
        self._time_backward_input: float | None = None
        self._time_backward_kernel: float | None = None
        # TODO: ^ remove
    def _init_and_check_shapes(self) -> tuple[tuple[int, ...], int]:
        # TODO: maybe also ensure that padding is not too big
        assert len(self._kernel_size) == len(self._padding), f"kernel size {self._kernel_size} incompatible with padding {self._padding}"
        assert len(self._kernel_size) == len(self._stride)
        input_shape = self._deps[0].get_shape()
        non_spatial_dims = len(input_shape) - len(self._kernel_size) - 1 # the -1 is the channels dimension
        assert non_spatial_dims >= 0, \
            f"number of dimensions of kernel with size {self._kernel_size} is too big for input of shape {input_shape}"
        assert all(
            input_shape[non_spatial_dims + 1 + i] + 2 * self._padding[i] >= self._kernel_size[i]
            for i in range(len(self._kernel_size))
        ), f"kernel of size {self._kernel_size} is too big for input of shape {input_shape}!"
        return input_shape, non_spatial_dims
    @override
    def get_shape(self):
        return self._input_shape[:self._non_spatial_dims] + (self._out_channels,) + tuple(
            1 + (self._input_shape[self._non_spatial_dims + 1 + i] + 2 * self._padding[i] - self._kernel_size[i]) // self._stride[i]
            for i in range(len(self._kernel_size))
        )
    @override
    def _get_value(self):
        depval = self._input_node.get_value()

        t0 = time.time() # TODO: remove
        self._depval_padded = utils.pad_lr(depval, self._padding, 0.0)
        self._padded_input_shape = self._depval_padded.shape
        kerval = self._kernels_node.get_value()
        result = self._multichannel_convolution_function(self._depval_padded, kerval, self._stride)

        self._time_forward = time.time() - t0 # TODO: remove
        return result
    @override
    def _get_input_gradients(self, output_gradient: np.ndarray):
        _val = self.get_value()
        assert self._padded_input_shape is not None
        assert self._depval_padded is not None

        kerval = self._kernels_node.get_value()

        t0 = time.time() # TODO: remove
        input_padded_grad_possibly_smaller = self._multichannel_transposed_convolution_functoion(output_gradient, kerval, self._stride)
        input_padded_grad = utils.pad_r(
            input_padded_grad_possibly_smaller,
            tuple(
                self._padded_input_shape[i] - input_padded_grad_possibly_smaller.shape[i]
                for i in range(self._non_spatial_dims, len(self._padded_input_shape))
            ),
            0.0
        )
        input_grad = utils.unpad_lr(input_padded_grad, self._padding)
        self._time_backward_input = time.time() - t0 # TODO: remove

        t0 = time.time() # TODO: remove
        ker_grad = np.zeros(kerval.shape)
        for idx in itertools.product(*[range(k) for k in self._kernel_size]):
            indexer = (slice(0, None),) * (self._non_spatial_dims + 1) + tuple(
                slice(
                    idx[meta_i],
                    idx[meta_i] + output_gradient.shape[self._non_spatial_dims + meta_i + 1] * self._stride[meta_i],
                    self._stride[meta_i]
                )
                for meta_i in range(len(idx))
            )
            arr = np.tensordot(
                output_gradient, self._depval_padded[indexer],
                axes=(
                    tuple(i for i in range(len(output_gradient.shape)) if i != self._non_spatial_dims),
                    tuple(i for i in range(len(output_gradient.shape)) if i != self._non_spatial_dims),
                    # ^ non_spatial_dims is not only the number of non-spatial dimensions, but also the index of the channels dimension
                    # essentially: for each kernel placement & point in kernel, we compute the outer product of the corresponding vectors of channel values
                )
            )
            ker_grad[:,:,*idx] = arr
        self._time_backward_kernel = time.time() - t0 # TODO: remove
        
        return [input_grad, ker_grad]
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_multichannel_convolution_node(self)
    @override
    def __repr__(self):
        return f"MultichannelConvolutionNode({self._input_node}, {self._kernels_node}, {self._padding}, {self._stride})"