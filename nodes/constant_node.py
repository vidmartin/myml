
from __future__ import annotations
from typing import *
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode

TResult = TypeVar("TResult")

class ConstantNode(TensorNode):
    def __init__(self, value: np.ndarray, name: str | None = None):
        self._value = value
        self._deps: list[TensorNode] = []
        self._ders: list[np.ndarray] = []
        self._name = name
    
    @override
    def get_value(self):
        return self._value
    @override
    def get_shape(self):
        return tuple(self._value.shape)
    @override
    def get_direct_dependencies(self):
        return self._deps
    @override
    def get_leaf_dependencies(self):
        return { self }
    @override
    def __repr__(self):
        return f"ConstantNode({"\"" + self._name + "\"" if self._name is not None else "<anon>"})"
    
    @override
    def get_gradients_against(self, leaves: list[ConstantNode], output_gradient: np.ndarray | None = None):
        return [
            output_gradient if leaf is self else np.zeros(leaf.get_shape())
            for leaf in leaves
        ]
    @override
    def _get_input_gradients(self, output_gradient):
        return []
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_constant_node(self)
    