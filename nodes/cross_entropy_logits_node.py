
from __future__ import annotations
from typing import *
import elementwise
import permutation
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.constant_node import ConstantNode
from nodes.wrapping_node import WrappingNode
from nodes.elementwise_node import ElementwiseNode
from nodes.sum_node import SumNode
from nodes.transpose_node import TransposeNode
from nodes.logsumexp_node import LogSumExpNode

TResult = TypeVar("TResult")

class CrossEntropyLogitsNode(WrappingNode):
    """
    Calculate cross entropy loss along the last dimension, where `yhat_logits` are predicted *logits* and `y_probas` are the target *probabilities*.

    Use together with `TransposeNode` to perform CrossEntropyLogits along any dimension.
    """
    def __init__(self, yhat_logits: TensorNode, y_probas: TensorNode):
        dep_shape = yhat_logits.get_shape()
        assert dep_shape == y_probas.get_shape()
        self._yhat = yhat_logits
        self._y = y_probas
        self._shape = dep_shape[:-1]

        def construct(nodes: list[ConstantNode]):
            yhat_node, y_node = nodes

            mul_node = ElementwiseNode(elementwise.ElementwiseMul(2), [yhat_node, y_node])
            sum_node = SumNode(TransposeNode(mul_node, permutation.Permutation.bring_to_front((-1,), len(dep_shape))), 1)
            neg_sum_node = ElementwiseNode(elementwise.ElementwiseScale(-1.0), [sum_node])

            logsumexp_node = LogSumExpNode(yhat_node)
            sum_probas_node = SumNode(TransposeNode(y_node, permutation.Permutation.bring_to_front((-1,), len(dep_shape))), 1)
            logsumexp_mul_node = ElementwiseNode(elementwise.ElementwiseMul(2), [logsumexp_node, sum_probas_node])
            # ^ probabilities should add up to one (when used correctly), so the multiplication by sum_probas_node is redundant in forward pass
            # however, it is important for computation of gradient against y_node
            # however, in practical use cases one rarely needs to compute that gradient
            # CONSIDER: make the sum_probas_node & logsumexp_mul_node optional

            add_node = ElementwiseNode(elementwise.ElementwiseAdd(2), [neg_sum_node, logsumexp_mul_node])
            return add_node

        super().__init__([yhat_logits, y_probas], construct)
    @override
    def __repr__(self):
        return f"CrossEntropyLogitsNode({self._yhat}, {self._y})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_cross_entropy_logits_node(self)
