
from __future__ import annotations
from typing import *
import numpy as np
import itertools
from nodes.lazy_dependent_node import LazyDependentNode
from nodes.tensor_node import TensorNode
from node_visitor.node_visitor import NodeVisitor
import utils

TResult = TypeVar("TResult")

class SliceNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, slice_dim: int, slice_index: int):
        super().__init__([dep])
        self._slice_dim = slice_dim
        self._slice_index = slice_index
    @override
    def get_shape(self):
        depshape = self._deps[0].get_shape()
        return tuple(
            v for i, v in enumerate(depshape)
            if i != self._slice_dim
        )
    def _get_indexer(self):
        return tuple(
            self._slice_index if i == self._slice_dim else slice(0, None)
            for i in range(len(self._deps[0].get_shape()))
        )
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return depval[self._get_indexer()]
    @override
    def _get_input_gradients(self, output_gradient):
        result = np.zeros(self._deps[0].get_shape())
        result[self._get_indexer()] = output_gradient
        return [result]
    @override
    def accept(self, visitor: NodeVisitor[TResult]):
        raise NotImplementedError() # TODO: add corresponding visitor method and call it here
    @override
    def __repr__(self):
        return f"SliceNode({self._deps[0]}, {self._slice_dim}, {self._slice_index})"
