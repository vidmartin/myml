
from __future__ import annotations
from typing import *
import itertools
import numpy as np
from nodes import LazyDependentNode
from nodes.tensor_node import TensorNode
from node_visitor.node_visitor import NodeVisitor

TResult = TypeVar("TResult")

# TODO: dilation

class MaxPoolNode(LazyDependentNode):
    def __init__(
        self,
        dep: TensorNode,
        kernel_size: tuple[int, ...],
        padding: tuple[int, ...],
        stride: tuple[int, ...]
    ):
        super().__init__([dep])
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self._init_and_check_shapes()
    def _init_and_check_shapes(self):
        # TODO: maybe also ensure that padding is not too big
        assert len(self._kernel_size) == len(self._padding)
        assert len(self._kernel_size) == len(self._stride)
        self._input_shape = self._deps[0].get_shape()
        self._non_spatial_dims = len(self._input_shape) - len(self._kernel_size)
        assert self._non_spatial_dims >= 0, \
            f"number of dimensions of kernel with size {self._kernel_size} is too big for input of shape {input_shape}"
        assert all(
            self._input_shape[self._non_spatial_dims + i] + 2 * self._padding[i] >= self._kernel_size[i]
            for i in range(len(self._kernel_size))
        ), f"kernel of size {self._kernel_size} is too big for input of shape {self._input_shape}!"
    @override
    def get_shape(self):
        return self._input_shape[:self._non_spatial_dims] + tuple(
            1 + (self._input_shape[self._non_spatial_dims + i] + 2 * self._padding[i] - self._kernel_size[i]) // self._stride[i]
            for i in range(len(self._kernel_size))
        )
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        depval_padded = np.full(
            self._input_shape[:self._non_spatial_dims] \
            + tuple(
                v + 2 * self._padding[i]
                for i, v in enumerate(self._input_shape[self._non_spatial_dims:])
            ),
            -np.inf
        )
        depval_padded_indexer = (slice(0, None),) * self._non_spatial_dims + tuple(
            slice(p, -p) for p in self._padding
        )
        depval_padded[depval_padded_indexer] = depval
        
        shape = self.get_shape()
        temp = np.full(shape, -np.inf)
        for idx in itertools.product(*[range(k) for k in self._kernel_size]):
            indexer = (slice(0, None),) * self._non_spatial_dims + tuple(
                slice(i, depval_padded.shape[self._non_spatial_dims + meta_i] - self._kernel_size[meta_i] + i + 1, self._stride[meta_i])
                for meta_i, i in enumerate(idx)
            )
            # TODO: ^ understand why this works (especially the +1)
            depval_padded_indexed = depval_padded[indexer]
            temp = np.maximum(temp, depval_padded_indexed)
        return temp
    @override
    def _get_input_gradients(self, output_gradient):
        return super()._get_input_gradients(output_gradient)
    @override
    def accept(self, visitor: NodeVisitor[TResult]):
        return super().accept(visitor)
    @override
    def __repr__(self):
        return f"MaxPoolNode({self._deps[0]}, {self._kernel_size}, {self._padding}, {self._stride})"