
from __future__ import annotations
from typing import *
import numpy as np
import utils
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class SoftmaxNode(LazyDependentNode):
    """
    Perform softmax over the last dimension.

    Use together with `TransposeNode` to perform softmax along any dimension.
    """
    def __init__(self, dep: TensorNode):
        super().__init__([dep])
    @override
    def get_shape(self):
        return self._deps[0].get_shape()
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return utils.softmax(depval)
    @override
    def _get_input_gradients(self, output_gradient):
        depval = self._deps[0].get_value()
        softmaxed = utils.softmax(depval)
        temp = output_gradient * softmaxed
        return [temp - softmaxed * temp.sum(-1)[...,np.newaxis]]
    @override
    def __repr__(self):
        return f"SoftmaxNode({self._deps[0]})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_softmax_node(self)
