
from __future__ import annotations
from typing import *
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class SumNode(LazyDependentNode):
    """
    Sum along the first `n_axes_to_sum` dimensions.
    
    Use together with `TransposeNode` to get sum along any dimension.
    """
    def __init__(self, dep: TensorNode, n_axes_to_sum: int):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        assert n_axes_to_sum <= len(dep_sh)
        self._shape = dep_sh[n_axes_to_sum:]
        self._n_axes_to_sum = n_axes_to_sum
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return depval.sum(tuple(range(self._n_axes_to_sum)))
    @override
    def _get_input_gradients(self, output_gradient):
        dep_sh = self._deps[0].get_shape()
        indexer = (np.newaxis,) * self._n_axes_to_sum + (slice(0, None),) * len(self.get_shape())
        return [output_gradient[indexer] + np.zeros(dep_sh)]
    @override
    def __repr__(self):
        return f"SumNode({self._deps[0]}, {self._n_axes_to_sum})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_sum_node(self)
