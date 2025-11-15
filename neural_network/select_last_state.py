
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode, ParameterSpecification
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise

TResult = TypeVar("TResult")

class SelectLastStateModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(self, sequence_length_metadata_key: str | None):
        """
        `sequence_length_metadata_key`: if a string key is given, the corresponding metadata value will be used
        as information about how long the individual sequences in the batch are. It will expect
        the metadata value to be an array of integers with length corresponding to batch size. If set to
        `None`, the length of all the sequences is taken to be the size of the input tensor along the corresponding dimension.
        """
        self._sequence_length_metadata_key = sequence_length_metadata_key
    @override
    def get_params(self):
        return {}
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        input_shape = input.get_shape()
        assert len(input_shape) == 3, f"only 3-dimensional arrays are supported by SelectLastStateModule"

        if self._sequence_length_metadata_key is not None:
            output_node = nodes.SelectNode(input, 0, 1, params[self._sequence_length_metadata_key])
            return ComputationalGraph(output_node, {})
        else:
            output_node = nodes.SliceNode(input, len(input_shape) - 2, input_shape[-2] - 1)
            return ComputationalGraph(output_node, {})
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        raise NotImplementedError() # TODO
