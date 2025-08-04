
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode
import nodes

class InputNumpyModule(NeuralNetwork[np.ndarray]):
    def __init__(self, wrapped: NeuralNetwork[nodes.TensorNode]):
        self._wrapped = wrapped
    @override
    def get_params(self):
        return self._wrapped.get_params()
    @override
    def _construct(self, input: np.ndarray, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        input_node = nodes.ConstantNode(input)
        return self._wrapped.construct(input_node, params, mode)
