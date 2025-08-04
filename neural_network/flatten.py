
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode
import nodes

class FlattenModule(NeuralNetwork[nodes.TensorNode]):
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        shape = input.get_shape()
        return ComputationalGraph(
            output_node=nodes.ReshapeNode(input, (shape[0], -1)),
            param_nodes={},
        )
