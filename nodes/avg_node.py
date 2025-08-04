
from __future__ import annotations
from typing import *
import functools
import elementwise
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.constant_node import ConstantNode
from nodes.wrapping_node import WrappingNode
from nodes.sum_node import SumNode
from nodes.elementwise_node import ElementwiseNode

TResult = TypeVar("TResult")
    
class AvgNode(WrappingNode):
    """
    Computes the average (aka mean) along the first dimension.

    Use together with `TransposeNode` to compute average along any dimension.
    """
    def __init__(self, dep: TensorNode, n_axes_to_avg: int):
        dep_shape = dep.get_shape()
        def construct(nodes: list[ConstantNode]):
            input_, = nodes
            sum_node = SumNode(input_, n_axes_to_avg)
            divisor = functools.reduce(lambda a, b: a * b, dep_shape[:n_axes_to_avg])
            scale_node = ElementwiseNode(elementwise.ElementwiseScale(1.0 / divisor), [sum_node])
            return scale_node
        self._n_axes_to_avg = n_axes_to_avg # not used here but may be used by visitors (we allow visitors to inspect private attrs)
        super().__init__([dep], construct)
    @override
    def __repr__(self):
        return f"AvgNode({self._deps[0]})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_avg_node(self)
