
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode, ParameterSpecification
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise

TResult = TypeVar("TResult")

INPUT_WEIGHT_PARAM_NAME = "input_weight"
STATE_WEIGHT_PARAM_NAME = "state_weight"
INPUT_BIAS_PARAM_NAME = "input_bias"
STATE_BIAS_PARAM_NAME = "state_bias"
INITIAL_STATE_PARAM_NAME = "initial_state"

class RNNModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(
        self,
        activation: elementwise.ElementwiseUnary,
        input_dim: int,
        state_dim: int,
        parametrize_initial_state: bool = False,
    ):
        self._activation = activation
        self._input_dim = input_dim
        self._state_dim = state_dim
        self._parametrize_initial_state = parametrize_initial_state
    @override
    def get_params(self):
        return {
            INPUT_WEIGHT_PARAM_NAME: ParameterSpecification((self._input_dim, self._state_dim)),
            STATE_WEIGHT_PARAM_NAME: ParameterSpecification((self._state_dim, self._state_dim)),
            INPUT_BIAS_PARAM_NAME: ParameterSpecification((self._state_dim,)),
            STATE_BIAS_PARAM_NAME: ParameterSpecification((self._state_dim,)),
        } | ({
            INITIAL_STATE_PARAM_NAME: ParameterSpecification((self._state_dim,))
        } if self._parametrize_initial_state else {})
    def _get_initial_state(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> tuple[nodes.TensorNode, dict[str, nodes.TensorNode]]:
        input_shape = input.get_shape()
        if self._parametrize_initial_state:
            param = nodes.ConstantNode(params[INITIAL_STATE_PARAM_NAME])
            return nodes.ExtendNode(param, input_shape[:-2]), { INITIAL_STATE_PARAM_NAME: param }
        state_shape = input_shape[:-2] + (self._state_dim,)
        return nodes.ConstantNode(np.zeros(state_shape)), {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        input_shape = input.get_shape()
        sequence_length = input_shape[-2] # input shape: (..., sequence length, sequence item vector)

        initial_state, initial_state_param_dict = self._get_initial_state(input, params, mode)
        states: list[nodes.TensorNode] = [initial_state]

        input_weight_param = nodes.ConstantNode(params[INPUT_WEIGHT_PARAM_NAME])
        state_weight_param = nodes.ConstantNode(params[STATE_WEIGHT_PARAM_NAME])
        input_bias_param = nodes.ConstantNode(params[INPUT_BIAS_PARAM_NAME])
        state_bias_param = nodes.ConstantNode(params[STATE_BIAS_PARAM_NAME])

        for i in range(sequence_length):
            item_input = nodes.SliceNode(input, len(input_shape) - 2, i)
            input_contrib = nodes.ElementwiseNode(elementwise.ElementwiseAdd(2), [
                nodes.TensorDotNode(item_input, input_weight_param, 1),
                nodes.ExtendNode(input_bias_param, input_shape[:-2])
            ])
            state_contrib = nodes.ElementwiseNode(elementwise.ElementwiseAdd(2), [
                nodes.TensorDotNode(states[-1], state_weight_param, 1),
                nodes.ExtendNode(state_bias_param, input_shape[:-2])
            ])
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
                INPUT_BIAS_PARAM_NAME: input_bias_param,
                STATE_BIAS_PARAM_NAME: state_bias_param,
            } | initial_state_param_dict
        )
            
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        raise NotImplementedError() # TODO