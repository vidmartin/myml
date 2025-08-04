
from __future__ import annotations
from typing import *
import numpy as np
import utils
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class LogSumExpNode(LazyDependentNode):
    """
    Perform the LogSumExp operation along the last dimension.

    Use together with `TransposeNode` to perform LogSumExp along any dimension.
    """
    def __init__(self, dep: TensorNode):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        self._shape = dep_sh[:-1]
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return utils.log_sum_exp(depval)
    @override
    def _get_input_gradients(self, output_gradient):
        depval = self._deps[0].get_value()
        softmaxed = utils.softmax(depval)
        return [output_gradient[...,np.newaxis] * softmaxed]
    @override
    def __repr__(self):
        return f"LogSumExpNode({self._deps[0]})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_logsumexp_node(self)