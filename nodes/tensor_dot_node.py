
from __future__ import annotations
from typing import *
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class TensorDotNode(LazyDependentNode):
    def __init__(
        self,
        lhs: TensorNode,
        rhs: TensorNode,
        n_axes_to_contract: int,
    ):
        super().__init__([lhs, rhs])

        lhs_sh = lhs.get_shape()
        rhs_sh = rhs.get_shape()
        assert len(lhs_sh) >= n_axes_to_contract
        assert len(rhs_sh) >= n_axes_to_contract
        assert all(lhs_sh[-n_axes_to_contract + i] == rhs_sh[i] for i in range(n_axes_to_contract)), \
            f"the shapes of the inputs aren't compatible with TensorDotNode; lhs shape is {lhs_sh}, rhs shape is {rhs_sh}"

        self._lhs = lhs
        self._rhs = rhs
        self._n_axes_to_contract = n_axes_to_contract
        self._shape = lhs_sh[:-n_axes_to_contract] + rhs_sh[n_axes_to_contract:]
    
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self) -> np.ndarray:
        lhs_v = self._lhs.get_value()
        rhs_v = self._rhs.get_value()
        return np.tensordot(lhs_v, rhs_v, self._n_axes_to_contract)
    
    @override
    def _get_input_gradients(self, output_gradient):
        lhs_sh, rhs_sh = self._lhs.get_shape(), self._rhs.get_shape()
        n_lhs_free_axes, n_rhs_free_axes = \
            len(lhs_sh) - self._n_axes_to_contract, \
            len(rhs_sh) - self._n_axes_to_contract
        lhs_v, rhs_v = self._lhs.get_value(), self._rhs.get_value()
        return [
            np.tensordot(
                output_gradient,
                rhs_v.transpose(
                    tuple(range(self._n_axes_to_contract, len(rhs_sh))) + \
                    tuple(range(self._n_axes_to_contract))
                ),
                n_rhs_free_axes
            ),
            np.tensordot(
                lhs_v.transpose(
                    tuple(range(n_lhs_free_axes, len(lhs_sh))) + \
                    tuple(range(n_lhs_free_axes))
                ),
                output_gradient,
                n_lhs_free_axes
            )
        ]
    
    @override
    def __repr__(self):
        return f"TensorDotNode({self._lhs}, {self._rhs}, {self._n_axes_to_contract})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_tensor_dot_node(self)
