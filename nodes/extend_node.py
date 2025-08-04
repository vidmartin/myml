
from __future__ import annotations
from typing import *
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class ExtendNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, prepend_dims: tuple[int, ...]):
        super().__init__([dep])
        self._prepend_dims = prepend_dims
    @override
    def get_shape(self):
        return self._prepend_dims + self._deps[0].get_shape()
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        indexer = (np.newaxis,) * len(self._prepend_dims) + (slice(0, None),) * len(depval.shape)
        dummy = np.zeros(self._prepend_dims + (1,) * len(depval.shape))
        return depval[indexer] + dummy
    @override
    def _get_input_gradients(self, output_gradient):
        return [np.sum(output_gradient, tuple(range(len(self._prepend_dims))))]
    @override
    def __repr__(self):
        return f"ExtendNode({self._deps[0]}, {self._prepend_dims})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_extend_node(self)
