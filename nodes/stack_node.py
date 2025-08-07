
from __future__ import annotations
from typing import *
import numpy as np
from nodes.lazy_dependent_node import LazyDependentNode
from nodes.tensor_node import TensorNode
from node_visitor.node_visitor import NodeVisitor

TResult = TypeVar("TResult")

class StackNode(LazyDependentNode):
    def __init__(
        self,
        nodes_to_stack: list[TensorNode],
        stack_dim: int,
    ):
        assert nodes_to_stack, "`nodes_to_stack` cannot be empty"
        node_shapes = set(node.get_shape() for node in nodes_to_stack)
        assert len(node_shapes) == 1, \
            "all nodes in `nodes_to_stack` must have the same shape"
        super().__init__(nodes_to_stack)
        self._depshape = next(iter(node_shapes))
        self._stack_dim = self._fix_stack_dim(stack_dim, self._depshape)
    def _fix_stack_dim(self, stack_dim: int, depshape: tuple[int, ...]):
        if stack_dim < 0:
            stack_dim += len(depshape) + 1
        assert stack_dim >= 0 and stack_dim < len(depshape) + 1
        return stack_dim
    @override
    def get_shape(self):
        return self._depshape[:self._stack_dim] + (len(self._deps),) + self._depshape[self._stack_dim:]
    @override
    def _get_value(self):
        if len(self._deps) == 1:
            return self._deps[0].get_value()
        depvals = [node.get_value() for node in self._deps]
        return np.stack(arrays=depvals, axis=self._stack_dim)
    @override
    def _get_input_gradients(self, output_gradient):
        slicer_fn = lambda i: \
            (slice(0, None),) * self._stack_dim + \
            (i,) + \
            (slice(0, None),) * (len(self._depshape) - self._stack_dim)
        return [
            output_gradient[slicer_fn(i)] for i in range(len(self._deps))
        ]
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_stack_node(self)
    @override
    def __repr__(self):
        return f"StackNode({self._deps}, {self._stack_dim})"
