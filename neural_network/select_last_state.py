
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode, ParameterSpecification
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise

TResult = TypeVar("TResult")

class SelectLastStateModule(NeuralNetwork[nodes.TensorNode]):
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        input_shape = input.get_shape()
        output_node = nodes.SliceNode(input, len(input_shape) - 2, input_shape[-2] - 1)
        # TODO: a way to select different item for each datapoint
        return ComputationalGraph(output_node, {})
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        raise NotImplementedError() # TODO
