
from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
import numpy as np
from nodes.tensor_node import TensorNode
from nodes.constant_node import ConstantNode

TResult = TypeVar("TResult")

class LazyDependentNode(TensorNode):
    def __init__(self, deps: list[TensorNode]):
        self._value: np.ndarray | None = None
        self._deps = deps
        self._ders: list[np.ndarray] | None = None
        self._leaf_dependencies: set[ConstantNode] | None = None

    @override
    @final
    def get_value(self):
        if self._value is None:
            self._value = self._get_value()
        return self._value
    @abstractmethod
    def _get_value(self) -> np.ndarray:
        raise NotImplementedError()
    
    @override
    @final
    def get_direct_dependencies(self):
        return self._deps
    @override
    @final
    def get_leaf_dependencies(self):
        if self._leaf_dependencies is None:
            self._leaf_dependencies = self._get_leaf_dependencies()
        return self._leaf_dependencies
    def _get_leaf_dependencies(self):
        return set().union(*[
            dep.get_leaf_dependencies()
            for dep in self.get_direct_dependencies()
        ])