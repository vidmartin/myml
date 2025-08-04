
from __future__ import annotations
from typing import *
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode
from nodes.constant_node import ConstantNode

TResult = TypeVar("TResult")

class WrappingNode(LazyDependentNode):
    def __init__(self, deps: list[TensorNode], wrapper: Callable[[list[ConstantNode]], TensorNode]):
        super().__init__(deps)
        self._wrapper = wrapper

        self._initialized = False
        self._leaves: list[ConstantNode] | None = None
        self._wrapped: TensorNode | None = None
    def _initialize(self):
        assert not self._initialized
        self._leaves = [
            ConstantNode(dep.get_value())
            for dep in self.get_direct_dependencies()
        ]
        self._wrapped = self._wrapper(self._leaves)
        self._initialized = True
    @override
    def get_shape(self):
        if not self._initialized:
            self._initialize()
        assert self._wrapped is not None
        return self._wrapped.get_shape()
    @override
    def _get_value(self):
        if not self._initialized:
            self._initialize()
        assert self._wrapped is not None
        return self._wrapped.get_value()
    @override
    def _get_input_gradients(self, output_gradient):
        if not self._initialized:
            self._initialize()
        assert self._wrapped is not None
        assert self._leaves is not None
        return self._wrapped.get_gradients_against(self._leaves, output_gradient)
    @override
    def __repr__(self):
        if not self._initialized:
            return f"WrappingNode({self._deps}, <lambda>) {{ unitialized }}"
        return f"WrappingNode({self._deps}, <lambda>) {{ {self._wrapped} }}"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        raise NotImplementedError()
    