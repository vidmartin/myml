
from __future__ import annotations
from typing import *
import numpy as np
import itertools
from nodes.lazy_dependent_node import LazyDependentNode
from nodes.tensor_node import TensorNode
from node_visitor.node_visitor import NodeVisitor
import utils

TResult = TypeVar("TResult")

# TODO: dilation

class ConvolutionNode(LazyDependentNode):
    def __init__(
        self,
        input_node: TensorNode,
        kernel_node: TensorNode,
        padding: tuple[int, ...],
        stride: tuple[int, ...]
    ):
        super().__init__([input_node, kernel_node])
        self._input_node = input_node
        self._kernel_node = kernel_node
        self._kernel_size = self._kernel_node.get_shape()
        self._padding = padding
        self._stride = stride
        self._input_shape, self._non_spatial_dims = self._init_and_check_shapes()
        self._padded_input_shape: tuple[int, ...] | None = None
    def _init_and_check_shapes(self) -> tuple[tuple[int, ...], int]:
        # TODO: maybe also ensure that padding is not too big
        assert len(self._kernel_size) == len(self._padding)
        assert len(self._kernel_size) == len(self._stride)
        input_shape = self._deps[0].get_shape()
        non_spatial_dims = len(input_shape) - len(self._kernel_size)
        assert non_spatial_dims >= 0, \
            f"number of dimensions of kernel with size {self._kernel_size} is too big for input of shape {input_shape}"
        assert all(
            input_shape[non_spatial_dims + i] + 2 * self._padding[i] >= self._kernel_size[i]
            for i in range(len(self._kernel_size))
        ), f"kernel of size {self._kernel_size} is too big for input of shape {input_shape}!"
        return input_shape, non_spatial_dims
    @override
    def get_shape(self):
        return self._input_shape[:self._non_spatial_dims] + tuple(
            1 + (self._input_shape[self._non_spatial_dims + i] + 2 * self._padding[i] - self._kernel_size[i]) // self._stride[i]
            for i in range(len(self._kernel_size))
        )
    @override
    def _get_value(self):
        depval = self._input_node.get_value()
        depval_padded = utils.pad_lr(depval, self._padding, 0.0)
        self._padded_input_shape = depval_padded.shape
        kerval = self._kernel_node.get_value()
        return utils.convolution(depval_padded, kerval, self._stride)
    @override
    def _get_input_gradients(self, output_gradient: np.ndarray):
        _val = self.get_value()
        assert self._padded_input_shape is not None

        kerval = self._kernel_node.get_value()
        input_grad_possibly_smaller = utils.transposed_convolution(output_gradient, kerval, self._stride)
        input_padded_grad = utils.pad_r(
            input_grad_possibly_smaller,
            tuple(
                self._padded_input_shape[i] - input_grad_possibly_smaller.shape[i]
                for i in range(self._non_spatial_dims, len(self._padded_input_shape))
            ),
            0.0
        )
        input_grad = utils.unpad_lr(input_padded_grad, self._padding)
        return [input_grad, None]
    @override
    def accept(self, visitor: NodeVisitor[TResult]):
        raise NotImplementedError() # TODO: add convolution node method to visitor and call it here
    @override
    def __repr__(self):
        return f"ConvolutionNode({self._input_node}, {self._kernel_node}, {self._padding}, {self._stride})"