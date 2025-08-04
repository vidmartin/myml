
from typing import *
from abc import ABC, abstractmethod
import numpy as np
import nodes
from neural_network import ComputationalGraph, EvaluationMode

class LossFunction(ABC):
    @final
    def construct(self, graph: ComputationalGraph, target: np.ndarray, mode: EvaluationMode = EvaluationMode.INFERENCE) -> nodes.TensorNode:
        return self._construct(graph, target, mode)
    @abstractmethod
    def _construct(self, graph: ComputationalGraph, target: np.ndarray, mode: EvaluationMode) -> nodes.TensorNode:
        raise NotImplementedError()
