
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
import nodes
import elementwise
import utils

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
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        if mode == EvaluationMode.INFERENCE:
            return ComputationalGraph(
                output_node=input,
                param_nodes={}
            )
        elif mode == EvaluationMode.TRAINING:
            input_shape = input.get_shape()
            mask = (self._rng(input_shape[-1]) >= self._dropout_rate).astype(np.float32)
            mask_node = nodes.ConstantNode(mask)
            extended_mask_node = nodes.ExtendNode(mask_node, input_shape[:-1])
            return ComputationalGraph(
                output_node=nodes.ElementwiseNode(
                    function=elementwise.ElementwiseScale(1.0 / (1 - self._dropout_rate)),
                    deps=[
                        nodes.ElementwiseNode(
                            function=elementwise.ElementwiseMul(2),
                            deps=[extended_mask_node, input]
                        )
                    ]
                ),
                param_nodes={}
            )
        else:
            raise NotImplementedError(f"unknown evaluation mode {mode}")
