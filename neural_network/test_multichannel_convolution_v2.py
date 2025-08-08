
import itertools
from typing import *
import unittest
import numpy as np
import torch
import nodes
from neural_network.neural_network import EvaluationMode
from neural_network.multichannel_convolution_v2 import MultichannelConvolutionV2Module

class MultichannelConvolutionV2TestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(24434123)
    def test_multichannel_convolution_2d_no_bias(self):
        in_channels, out_channels = 10, 15
        kersize = (8, 9)
        stride = (2, 3)
        pad = (1, 1)
        input_arr = self._rng.standard_normal((50, in_channels, 100, 100))
        kernel_arrs = (self._rng.random((out_channels, in_channels, *kersize)) * 2 - 1) * (6.0 / sum(kersize)) ** 0.5

        torch_conv2d = torch.nn.Conv2d(in_channels, out_channels, kersize, stride, pad, bias=False)
        torch_conv2d.get_parameter("weight").data = torch.tensor(kernel_arrs, requires_grad=False)
        input_torch = torch.tensor(input_arr)
        output_torch: torch.Tensor = torch_conv2d(input_torch)
        sum_torch = output_torch.sum()
        sum_torch.backward()

        my_conv2d = MultichannelConvolutionV2Module(in_channels, out_channels, kersize, pad, stride)
        input_node = nodes.ConstantNode(input_arr)
        graph = my_conv2d.construct(
            input=input_node,
            params={ "kernels": kernel_arrs },
            mode=EvaluationMode.TRAINING
        )
        sum_node = nodes.SumNode(graph.output_node, 4)
        kernels_grad, = sum_node.get_gradients_against([graph.param_nodes["kernels"]])

        self.assertTrue(np.allclose(output_torch.detach().numpy(), graph.output_node.get_value()))
        self.assertTrue(np.allclose(
            kernels_grad,
            torch_conv2d.get_parameter("weight").grad.detach().numpy()
        ))
    def test_multichannel_convolution_2d_with_bias(self):
        in_channels, out_channels = 10, 15
        kersize = (8, 9)
        stride = (2, 3)
        pad = (1, 1)
        input_arr = self._rng.standard_normal((50, in_channels, 100, 100))
        kernel_arrs = (self._rng.random((out_channels, in_channels, *kersize)) * 2 - 1) * (6.0 / sum(kersize)) ** 0.5
        bias_arr = (self._rng.random((out_channels,)) * 2 - 1) * (6.0 / (out_channels + 1)) ** 0.5

        torch_conv2d = torch.nn.Conv2d(in_channels, out_channels, kersize, stride, pad, bias=True)
        torch_conv2d.get_parameter("weight").data = torch.tensor(kernel_arrs, requires_grad=False)
        torch_conv2d.get_parameter("bias").data = torch.tensor(bias_arr, requires_grad=False)
        input_torch = torch.tensor(input_arr)
        output_torch: torch.Tensor = torch_conv2d(input_torch)
        sum_torch = output_torch.sum()
        sum_torch.backward()

        my_conv2d = MultichannelConvolutionV2Module(in_channels, out_channels, kersize, pad, stride, bias=True)
        input_node = nodes.ConstantNode(input_arr)
        graph = my_conv2d.construct(
            input=input_node,
            params={ "kernels": kernel_arrs, "bias": bias_arr },
            mode=EvaluationMode.TRAINING
        )
        sum_node = nodes.SumNode(graph.output_node, 4)
        kernels_grad, = sum_node.get_gradients_against([graph.param_nodes["kernels"]])

        self.assertTrue(np.allclose(output_torch.detach().numpy(), graph.output_node.get_value()))
        self.assertTrue(np.allclose(
            kernels_grad,
            torch_conv2d.get_parameter("weight").grad.detach().numpy()
        ))

        bias_node_grad, = sum_node.get_gradients_against([graph.param_nodes["bias"]])
        self.assertTrue(np.allclose(bias_node_grad, torch_conv2d.get_parameter("bias").grad.detach().numpy(), atol=0.001))
