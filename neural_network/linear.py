
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
import elementwise
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes

TResult = TypeVar("TResult")

WEIGHT_PARAM_NAME = "weight"
BIAS_PARAM_NAME = "bias"

class LinearModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(self, in_features: int, out_features: int):
        self._in_features = in_features
        self._out_features = out_features
    @override
    def get_params(self):
        return {
            WEIGHT_PARAM_NAME: ParameterSpecification((self._in_features, self._out_features)),
            BIAS_PARAM_NAME: ParameterSpecification((self._out_features,)),
        }
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        shape = input.get_shape()
        weight_node = nodes.ConstantNode(params[WEIGHT_PARAM_NAME])
        bias_node = nodes.ConstantNode(params[BIAS_PARAM_NAME])
        return ComputationalGraph(
            output_node=nodes.ElementwiseNode(
                elementwise.ElementwiseAdd(2), [
                    nodes.ExtendNode(bias_node, (shape[0],)),
                    nodes.TensorDotNode(input, weight_node, 1),
                ]
            ),
            param_nodes={
                WEIGHT_PARAM_NAME: weight_node,
                BIAS_PARAM_NAME: bias_node,
            },
        )
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_linear(self)
