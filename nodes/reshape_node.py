
from __future__ import annotations
from typing import *
import utils
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class ReshapeNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, target_shape: tuple[int, ...]):
        dep_shape = dep.get_shape()
        super().__init__([dep])
        self._target_shape = utils.normalize_dest_shape(dep_shape, target_shape)
    @override
    def get_shape(self):
        return self._target_shape
    @override
    def _get_value(self):
        return self._deps[0].get_value().reshape(self._target_shape)
    @override
    def _get_input_gradients(self, output_gradient):
        return [output_gradient.reshape(self._deps[0].get_shape())]
    @override
    def accept(self, visitor):
        return visitor.visit_reshape_node(self)
    @override
    def __repr__(self):
        return f"ReshapeNode({self._deps[0]}, {self._target_shape})"
