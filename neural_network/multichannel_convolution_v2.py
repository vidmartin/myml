
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise
import permutation

TResult = TypeVar("TResult")

class MultichannelConvolutionV2Module(NeuralNetwork[nodes.TensorNode]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        padding: tuple[int, ...],
        stride: tuple[int, ...],
        bias: bool = False,
        multichannel_convolution_function_or_version: Callable[[np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray] | int = 1,
    ):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self._bias = bias
        self._multichannel_convolution_function_or_version = multichannel_convolution_function_or_version
    @override
    def get_params(self):
        kernels_dict = {
            "kernels": ParameterSpecification(
                (self._out_channels, self._in_channels, *self._kernel_size)
            )
        }
        if self._bias:
            return kernels_dict | {
                "bias": ParameterSpecification((self._out_channels,))
            }
        return kernels_dict
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        kernels_node = nodes.ConstantNode(params["kernels"])
        conv_node = nodes.MultichannelConvolutionNode(input, kernels_node, self._padding, self._stride, self._multichannel_convolution_function_or_version)
        out_shape = conv_node.get_shape()
        non_spatial_dims = len(out_shape) - len(self._kernel_size) - 1
        if not self._bias:
            return ComputationalGraph(
                output_node=conv_node,
                param_nodes={ "kernels": kernels_node }
            )

        bias_node = nodes.ConstantNode(params["bias"])
        return ComputationalGraph(
            output_node=nodes.ElementwiseNode(
                elementwise.ElementwiseAdd(2), [
                    conv_node,
                    nodes.TransposeNode(
                        nodes.ExtendNode(
                            bias_node,
                            out_shape[:non_spatial_dims] + out_shape[(non_spatial_dims + 1):]
                        ),
                        permutation.Permutation.bring_to_back(
                            tuple(non_spatial_dims + i for i in range(len(self._kernel_size))),
                            len(out_shape),
                        )
                    )
                ]
            ),
            param_nodes={
                "kernels": kernels_node,
                "bias": bias_node,
            }
        )
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_multichannel_convolution_v2(self)
