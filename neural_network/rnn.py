
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode, ParameterSpecification
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise

TResult = TypeVar("TResult")

INPUT_WEIGHT_PARAM_NAME = "input_weight"
STATE_WEIGHT_PARAM_NAME = "state_weight"

class RNNModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(
        self,
        activation: elementwise.ElementwiseUnary,
        input_dim: int,
        state_dim: int,
    ):
        self._activation = activation
        self._input_dim = input_dim
        self._state_dim = state_dim
    @override
    def get_params(self):
        return {
            INPUT_WEIGHT_PARAM_NAME: ParameterSpecification((self._input_dim, self._state_dim)),
            STATE_WEIGHT_PARAM_NAME: ParameterSpecification((self._state_dim, self._state_dim)),
        }
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        raise NotImplementedError()
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        raise NotImplementedError()