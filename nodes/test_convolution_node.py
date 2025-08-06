
from typing import *
import unittest
import numpy as np
import torch
import nodes

class ConvolutionNodeTestCase(unittest.TestCase):
    def test_convolution_shape(self):
        arr_1d = np.zeros((100,10,50))
        ker_size_1d = (13,)
        ker_1d = np.zeros(ker_size_1d)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d)
        ker_1d_torch = torch.tensor(ker_1d)
        res_1d_torch: torch.Tensor = torch.nn.functional.conv1d(
            arr_1d_torch,
            torch.stack((ker_1d_torch[torch.newaxis,...],) * 10),
            stride=stride_1d,
            padding=pad_1d,
            groups=10,
        )

        arr_1d_node = nodes.ConstantNode(arr_1d)
        ker_1d_node = nodes.ConstantNode(ker_1d)
        res_1d_node = nodes.ConvolutionNode(arr_1d_node, ker_1d_node, pad_1d, stride_1d)

        self.assertEqual(tuple(res_1d_torch.shape), res_1d_node.get_shape())

        arr_2d = np.zeros((100,10,50,50))
        ker_size_2d = (13,11)
        ker_2d = np.zeros(ker_size_2d)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d)
        ker_2d_torch = torch.tensor(ker_2d)
        res_2d_torch = torch.nn.functional.conv2d(
            arr_2d_torch,
            torch.stack((ker_2d_torch[torch.newaxis,...],) * 10),
            stride=stride_2d,
            padding=pad_2d,
            groups=10,
        )

        arr_2d_node = nodes.ConstantNode(arr_2d)
        ker_2d_node = nodes.ConstantNode(ker_2d)
        res_2d_node = nodes.ConvolutionNode(arr_2d_node, ker_2d_node, pad_2d, stride_2d)

        self.assertEqual(tuple(res_2d_torch.shape), res_2d_node.get_shape())

        arr_3d = np.zeros((100,10,50,50,50))
        ker_size_3d = (5,4,3)
        ker_3d = np.zeros(ker_size_3d)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d)
        ker_3d_torch = torch.tensor(ker_3d)
        res_3d_torch = torch.nn.functional.conv3d(
            arr_3d_torch,
            torch.stack((ker_3d_torch[torch.newaxis,...],) * 10),
            stride=stride_3d,
            padding=pad_3d,
            groups=10,
        )

        arr_3d_node = nodes.ConstantNode(arr_3d)
        ker_3d_node = nodes.ConstantNode(ker_3d)
        res_3d_node = nodes.ConvolutionNode(arr_3d_node, ker_3d_node, pad_3d, stride_3d)

        self.assertEqual(tuple(res_3d_torch.shape), res_3d_node.get_shape())

    def test_convolution_value(self):
        pass # TODO

    def test_convolution_derivative(self):
        pass # TODO
