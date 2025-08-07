
import itertools
from typing import *
import unittest
import numpy as np
import torch
import nodes
from neural_network.neural_network import EvaluationMode
from neural_network.multichannel_convolution import MultichannelConvolutionModule

class MultichannelConvolutionTestCase(unittest.TestCase):
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

        my_conv2d = MultichannelConvolutionModule(in_channels, out_channels, kersize, pad, stride)
        input_node = nodes.ConstantNode(input_arr)
        graph = my_conv2d.construct(
            input=input_node,
            params={
                f"ker_in{i}_out{o}": kernel_arrs[o,i,:,:]
                for o in range(out_channels)
                for i in range(in_channels)
            },
            mode=EvaluationMode.TRAINING
        )
        sum_node = nodes.SumNode(graph.output_node, 4)
        kernel_nodes_linear = [
            graph.param_nodes[f"ker_in{i}_out{o}"]
            for i in range(in_channels)
            for o in range(out_channels)
        ]
        kernel_nodes_grads = sum_node.get_gradients_against(kernel_nodes_linear)

        self.assertTrue(np.allclose(output_torch.detach().numpy(), graph.output_node.get_value()))

        for kernel_node_grad, (i, o) in zip(
            kernel_nodes_grads,
            itertools.product(range(in_channels), range(out_channels))
        ):
            self.assertTrue(np.allclose(
                kernel_node_grad,
                torch_conv2d.get_parameter("weight").grad.detach().numpy()[o,i,:,:]
            ))
