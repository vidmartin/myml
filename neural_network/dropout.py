
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise
import utils

TResult = TypeVar("TResult")

DROPOUT_RATE_PARAM_NAME = "dropout_rate"

class DropoutModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(
        self,
        dropout_rate: float,
        rng: utils.RandomGenerator | None = None,
    ):
        self._dropout_rate = dropout_rate
        self._rng = rng if rng is not None else utils.NumpyUniformRandomGenerator(None)
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        if mode == EvaluationMode.INFERENCE:
            return ComputationalGraph(
                output_node=input,
                param_nodes={}
            )
        elif mode == EvaluationMode.TRAINING:
            input_shape = input.get_shape()
            mask = (self._rng(input_shape) >= self._dropout_rate).astype(np.float32)
            mask_node = nodes.ConstantNode(mask)
            return ComputationalGraph(
                output_node=nodes.ElementwiseNode(
                    function=elementwise.ElementwiseScale(1.0 / (1 - self._dropout_rate)),
                    deps=[
                        nodes.ElementwiseNode(
                            function=elementwise.ElementwiseMul(2),
                            deps=[mask_node, input]
                        )
                    ]
                ),
                param_nodes={}
            )
        else:
            raise NotImplementedError(f"unknown evaluation mode {mode}")
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_dropout(self)
