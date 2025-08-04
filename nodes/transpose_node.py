
from __future__ import annotations
from typing import *
import permutation
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class TransposeNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, permutation: permutation.Permutation):
        dep_sh = dep.get_shape()
        assert len(permutation.permutation) == len(dep_sh)
        super().__init__([dep])
        self._permutation = permutation
        self._inverse_permutation = permutation.inverse()
        self._shape = tuple(dep_sh[permutation.permutation[i]] for i in range(len(dep_sh)))
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        return self._deps[0].get_value().transpose(self._permutation.permutation)
    @override
    def _get_input_gradients(self, output_gradient):
        return [output_gradient.transpose(self._inverse_permutation.permutation)]
    @override
    def __repr__(self):
        return f"TransposeNode({self._deps[0]}, {self._permutation})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_transpose_node(self)
