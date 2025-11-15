
from __future__ import annotations
from typing import *
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class SelectNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, select_axis: int, select_along_axis: int, indices: np.ndarray):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        assert select_axis < len(dep_sh)
        assert select_along_axis < len(dep_sh)
        assert select_along_axis != select_axis
        self._select_axis = select_axis
        self._select_along_axis = select_along_axis
        self._indices = indices
        self._shape = tuple(v for i, v in enumerate(dep_sh) if i != select_along_axis)
    @override
    def get_shape(self):
        return self._shape
    def _get_indexer(self):
        dep_sh = self._deps[0].get_shape()
        return tuple(
            np.arange(dep_sh[i]) if i == self._select_axis else (
                self._indices if i == self._select_along_axis else slice(0, None)
            ) for i in range(len(dep_sh))
        )
    @override
    def _get_value(self):
        indexer = self._get_indexer()
        dep_val = self._deps[0].get_value()
        return dep_val[indexer]
    @override
    def _get_input_gradients(self, output_gradient):
        grad = np.zeros(self._deps[0].get_shape())
        indexer = self._get_indexer()
        grad[indexer] = output_gradient
        return [grad]
    @override
    def __repr__(self):
        return f"SliceNode({self._deps[0]}, {self._select_axis}, {self._select_along_axis}, {self._indices})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_select_node(self)
