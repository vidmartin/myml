
from typing import *
import numpy as np
from elementwise import ElementwiseUnary
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes

TResult = TypeVar("TResult")

class ElementwiseModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(self, function: ElementwiseUnary):
        self._function = function
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        return ComputationalGraph(
            output_node=nodes.ElementwiseNode(self._function, [input]),
            param_nodes={},
        )
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_elementwise(self)
