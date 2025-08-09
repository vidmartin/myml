
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes

TResult = TypeVar("TResult")

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
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_flatten(self)
