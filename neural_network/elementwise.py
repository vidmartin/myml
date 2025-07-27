
from typing import *
import numpy as np
from elementwise import ElementwiseUnary
from neural_network.neural_network import NeuralNetwork, ComputationalGraph
import nodes

class ElementwiseModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(self, function: ElementwiseUnary):
        self._function = function
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray]) -> ComputationalGraph:
        return ComputationalGraph(
            output_node=nodes.ElementwiseNode(self._function, [input]),
            param_nodes={},
        )
