
from __future__ import annotations
from typing import *
import numpy as np
import elementwise
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode

TResult = TypeVar("TResult")

class ElementwiseNode(LazyDependentNode):
    def __init__(self, function: elementwise.ElementwiseFunction, deps: list[TensorNode]):
        super().__init__(deps)
        self._function = function

        shapes = {dep.get_shape() for dep in deps}
        assert len(shapes) == 1, f"cannot perform {function.get_input_count()}-ary elementwise function on tensors of differing shapes ({shapes})"
        # ^ we want to prevent broadcasting, that would require more careful calculation of derivatives ^
        # TODO: enable broadcasting
        self._shape: tuple[int, ...] = next(iter(shapes))
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        return self._function.evaluate_function([
            dep.get_value() for dep in self.get_direct_dependencies()
        ])
    @override
    def _get_input_gradients(self, output_gradient: np.ndarray):
        deps = self.get_direct_dependencies()
        dep_values = [dep.get_value() for dep in deps]
        return [
            output_gradient * self._function.evaluate_partial_derivative(i, dep_values)
            for i in range(len(deps))
        ]
    
    @override
    def __repr__(self):
        return f"ElementwiseNode({self._function}, [{', '.join(repr(dep) for dep in self.get_direct_dependencies())}])"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        return visitor.visit_elementwise_node(self)
