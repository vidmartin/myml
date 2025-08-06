
from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
import functools
import numpy as np
from node_visitor.node_visitor import NodeVisitor

if TYPE_CHECKING:
    from nodes.constant_node import ConstantNode

TResult = TypeVar("TResult")

class TensorNode(ABC):
    @abstractmethod
    def get_value(self) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_direct_dependencies(self) -> list[TensorNode]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_leaf_dependencies(self) -> set[ConstantNode]:
        raise NotImplementedError()
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
    
    def get_gradients_against(self, leaves: list[ConstantNode], output_gradient: np.ndarray | None = None):
        results = [np.zeros(leaf.get_shape()) for leaf in leaves]
        inputs_grads = self.get_input_gradients(output_gradient)
        for input_grad, input_ in zip(inputs_grads, self.get_direct_dependencies()):
            grads = input_.get_gradients_against(leaves, input_grad)
            for i in range(len(grads)):
                # vaguely: according to chain rule, gradients add up
                # TODO: understand more formally
                results[i] += grads[i]
        return results
    
    @final
    def get_input_gradients(self, output_gradient: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Returns "local gradients" of inputs given the gradient of the output
        in order corresponding to the return value of `get_direct_dependencies`.
        """
        shape = self.get_shape()
        if output_gradient is None:
            assert functools.reduce(lambda a, b: a * b, shape, 1) == 1, "`output_gradient` is optional only for nodes with scalar outputs"
            output_gradient = np.ones(shape, dtype=np.float32)
        assert output_gradient.shape == self.get_shape(), "`output_gradient` must have the same shape as the output of the given node"
        return self._get_input_gradients(output_gradient)
    @abstractmethod
    def _get_input_gradients(self, output_gradient: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError()
    
    @abstractmethod
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        raise NotImplementedError()
