
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
        # TODO: different ways to produce initial state
    @override
    def get_params(self):
        return {
            INPUT_WEIGHT_PARAM_NAME: ParameterSpecification((self._input_dim, self._state_dim)),
            STATE_WEIGHT_PARAM_NAME: ParameterSpecification((self._state_dim, self._state_dim)),
        }
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        input_shape = input.get_shape()
        sequence_length = input_shape[-2] # input shape: (..., sequence length, sequence item vector)

        state_shape = input_shape[:-2] + (self._state_dim,)
        states: list[nodes.TensorNode] = [nodes.ConstantNode(np.zeros(state_shape))]

        input_weight_param = nodes.ConstantNode(params[INPUT_WEIGHT_PARAM_NAME])
        state_weight_param = nodes.ConstantNode(params[STATE_WEIGHT_PARAM_NAME])

        for i in range(sequence_length):
            item_input = nodes.SliceNode(input, len(input_shape) - 2, i)
            input_contrib = nodes.TensorDotNode(item_input, input_weight_param, 1)
            state_contrib = nodes.TensorDotNode(states[-1], state_weight_param, 1)
            item_result = nodes.ElementwiseNode(self._activation, [
                nodes.ElementwiseNode(elementwise.ElementwiseAdd(2), [
                    input_contrib, state_contrib
                ])
            ])
            states.append(item_result)

        output_node = nodes.StackNode(states[1:], len(input_shape) - 2) # states[1:] to skip the initial state
        return ComputationalGraph(
            output_node, {
                INPUT_WEIGHT_PARAM_NAME: input_weight_param,
                STATE_WEIGHT_PARAM_NAME: state_weight_param,
            }
        )
            
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        raise NotImplementedError() # TODO