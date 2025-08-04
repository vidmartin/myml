
from __future__ import annotations
from typing import *
import numpy as np
from nodes import LazyDependentNode
from node_visitor.node_visitor import NodeVisitor

TResult = TypeVar("TResult")

class MaxPoolNode(LazyDependentNode):
    @override
    def get_shape(self):
        return super().get_shape()
    @override
    def _get_value(self):
        return super()._get_value()
    @override
    def _get_input_gradients(self, output_gradient):
        return super()._get_input_gradients(output_gradient)
    @override
    def accept(self, visitor: NodeVisitor[TResult]):
        return super().accept(visitor)