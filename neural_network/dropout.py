
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification
import nodes
import elementwise

DROPOUT_RATE_PARAM_NAME = "dropout_rate"

class DropoutModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(self, dropout_rate: float, rng: np.random.Generator):
        self._dropout_rate = dropout_rate
        self._rng = rng
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray]) -> ComputationalGraph:
        input_shape = input.get_shape()
        mask = self._rng.integers(0, 1, (input_shape[-1],), endpoint=True).astype(np.float32)
        mask_node = nodes.ConstantNode(mask)
        extended_mask_node = nodes.ExtendNode(mask_node, input_shape[:-1])
        return ComputationalGraph(
            output_node=nodes.ElementwiseNode(
                function=elementwise.ElementwiseMul(2),
                deps=[extended_mask_node, input]
            ),
            param_nodes={}
        )
