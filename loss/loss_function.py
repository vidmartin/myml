
from typing import *
import dataclasses
from abc import ABC, abstractmethod
import numpy as np
import nodes
from neural_network import ComputationalGraph

class LossFunction(ABC):
    @final
    def construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        return self._construct(graph, target)
    @abstractmethod
    def _construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        raise NotImplementedError()
