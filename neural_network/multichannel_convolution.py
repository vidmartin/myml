
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
import nodes
import elementwise
import utils

# TODO: dilation
# TODO: bias

class MultichannelConvolutionModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        padding: tuple[int, ...],
        stride: tuple[int, ...]
    ):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
    @override
    def get_params(self):
        return {
            f"ker_in{c_in}_out{c_out}": ParameterSpecification(
                self._kernel_size
            )
            for c_in in range(self._in_channels)
            for c_out in range(self._out_channels)
        }
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        input_shape = input.get_shape()
        channel_dim = len(input_shape) - len(self._kernel_size) - 1
        kernel_nodes = {
            key: nodes.ConstantNode(val)
            for key, val in params.items()
        }
        input_plane_nodes = [
            nodes.SliceNode(input, channel_dim, i)
            for i in range(self._in_channels)
        ]
        output_plane_nodes = [
            nodes.ElementwiseNode(
                elementwise.ElementwiseAdd(self._in_channels),
                [
                    nodes.ConvolutionNode(
                        input_node=input_plane_node,
                        kernel_node=kernel_nodes[f"ker_in{j}_out{i}"],
                        padding=self._padding,
                        stride=self._stride,
                    )
                    for j, input_plane_node in enumerate(input_plane_nodes)
                ]
            )
            for i in range(self._out_channels)
        ]
        output_node = nodes.StackNode(output_plane_nodes, channel_dim)
        return ComputationalGraph(
            output_node=output_node,
            param_nodes=kernel_nodes,
        )
