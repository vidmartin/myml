
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes

TResult = TypeVar("TResult")

class InputNumpyModule(NeuralNetwork[np.ndarray]):
    def __init__(self, wrapped: NeuralNetwork[nodes.TensorNode]):
        self._wrapped = wrapped
    @override
    def get_params(self):
        return self._wrapped.get_params()
    @override
    def _construct(self, input: np.ndarray, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        input_node = nodes.ConstantNode(input)
        return self._wrapped.construct(input_node, params, mode, metadata)
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_input_numpy(self)
