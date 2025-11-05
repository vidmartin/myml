
from __future__ import annotations
from typing import *
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class SelectNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, select_along_axis: int, select_axis: int, indices: np.ndarray):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        assert select_along_axis < len(dep_sh)
        assert select_axis < len(dep_sh)
        self._shape = tuple(v for i, v in enumerate(dep_sh) if i != select_axis)
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        raise NotImplementedError()
    @override
    def _get_input_gradients(self, output_gradient):
        raise NotImplementedError()
    @override
    def __repr__(self):
        raise NotImplementedError()
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        raise NotImplementedError()
