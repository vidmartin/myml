
from typing import *
from abc import ABC, abstractmethod
import unittest
import numpy as np
import torch
import nodes

class MultichannelConvolutionNodeCommonTests(ABC):
    def setUp(self):
        self._rng = np.random.default_rng(3124123)

    @abstractmethod
    def get_multichannel_convolution_version(self) -> int:
        raise NotImplementedError()

    def test_multichannel_convolution_shape(self):
        arr_1d = np.zeros((100,10,50))
        kers_size_1d = (11,10,13,)
        kers_1d = np.zeros(kers_size_1d)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d)
        kers_1d_torch = torch.tensor(kers_1d)
        res_1d_torch: torch.Tensor = torch.nn.functional.conv1d(
            arr_1d_torch,
            kers_1d_torch,
            stride=stride_1d,
            padding=pad_1d,
        )

        arr_1d_node = nodes.ConstantNode(arr_1d)
        kers_1d_node = nodes.ConstantNode(kers_1d)
        res_1d_node = nodes.MultichannelConvolutionNode(arr_1d_node, kers_1d_node, pad_1d, stride_1d, self.get_multichannel_convolution_version())

        self.assertEqual(tuple(res_1d_torch.shape), res_1d_node.get_shape())

        arr_2d = np.zeros((100,10,50,50))
        kers_size_2d = (12,10,13,11)
        kers_2d = np.zeros(kers_size_2d)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d)
        kers_2d_torch = torch.tensor(kers_2d)
        res_2d_torch = torch.nn.functional.conv2d(
            arr_2d_torch,
            kers_2d_torch,
            stride=stride_2d,
            padding=pad_2d,
        )

        arr_2d_node = nodes.ConstantNode(arr_2d)
        kers_2d_node = nodes.ConstantNode(kers_2d)
        res_2d_node = nodes.MultichannelConvolutionNode(arr_2d_node, kers_2d_node, pad_2d, stride_2d, self.get_multichannel_convolution_version())

        self.assertEqual(tuple(res_2d_torch.shape), res_2d_node.get_shape())

        arr_3d = np.zeros((100,10,50,50,50))
        kers_size_3d = (11,10,5,4,3)
        kers_3d = np.zeros(kers_size_3d)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d)
        kers_3d_torch = torch.tensor(kers_3d)
        res_3d_torch = torch.nn.functional.conv3d(
            arr_3d_torch,
            kers_3d_torch,
            stride=stride_3d,
            padding=pad_3d,
        )

        arr_3d_node = nodes.ConstantNode(arr_3d)
        kers_3d_node = nodes.ConstantNode(kers_3d)
        res_3d_node = nodes.MultichannelConvolutionNode(arr_3d_node, kers_3d_node, pad_3d, stride_3d, self.get_multichannel_convolution_version())

        self.assertEqual(tuple(res_3d_torch.shape), res_3d_node.get_shape())

    def test_multichannel_convolution_value(self):
        arr_1d = self._rng.standard_normal((100,10,50))
        kers_size_1d = (11,10,13)
        kers_1d = self._rng.standard_normal(kers_size_1d)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d)
        kers_1d_torch = torch.tensor(kers_1d)
        res_1d_torch: torch.Tensor = torch.nn.functional.conv1d(
            arr_1d_torch,
            kers_1d_torch,
            stride=stride_1d,
            padding=pad_1d,
        )

        arr_1d_node = nodes.ConstantNode(arr_1d)
        kers_1d_node = nodes.ConstantNode(kers_1d)
        res_1d_node = nodes.MultichannelConvolutionNode(arr_1d_node, kers_1d_node, pad_1d, stride_1d, self.get_multichannel_convolution_version())

        self.assertTrue(np.allclose(res_1d_torch.detach().numpy(), res_1d_node.get_value()))

        arr_2d = self._rng.standard_normal((100,10,50,50))
        kers_size_2d = (12,10,13,11)
        kers_2d = self._rng.standard_normal(kers_size_2d)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d)
        kers_2d_torch = torch.tensor(kers_2d)
        res_2d_torch = torch.nn.functional.conv2d(
            arr_2d_torch,
            kers_2d_torch,
            stride=stride_2d,
            padding=pad_2d,
        )

        arr_2d_node = nodes.ConstantNode(arr_2d)
        kers_2d_node = nodes.ConstantNode(kers_2d)
        res_2d_node = nodes.MultichannelConvolutionNode(arr_2d_node, kers_2d_node, pad_2d, stride_2d, self.get_multichannel_convolution_version())

        self.assertTrue(np.allclose(res_2d_torch.detach().numpy(), res_2d_node.get_value()))

        arr_3d = self._rng.standard_normal((100,10,50,50,50))
        kers_size_3d = (11,10,5,4,3)
        kers_3d = self._rng.standard_normal(kers_size_3d)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d)
        kers_3d_torch = torch.tensor(kers_3d)
        res_3d_torch = torch.nn.functional.conv3d(
            arr_3d_torch,
            kers_3d_torch,
            stride=stride_3d,
            padding=pad_3d,
        )

        arr_3d_node = nodes.ConstantNode(arr_3d)
        kers_3d_node = nodes.ConstantNode(kers_3d)
        res_3d_node = nodes.MultichannelConvolutionNode(arr_3d_node, kers_3d_node, pad_3d, stride_3d, self.get_multichannel_convolution_version())

        self.assertTrue(np.allclose(res_3d_torch.detach().numpy(), res_3d_node.get_value()))

    def test_multichannel_convolution_input_derivative(self):
        arr_1d = self._rng.standard_normal((100,10,50))
        kers_size_1d = (11,10,13,)
        kers_1d = self._rng.standard_normal(kers_size_1d)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d, requires_grad=True)
        kers_1d_torch = torch.tensor(kers_1d)
        res_1d_torch: torch.Tensor = torch.nn.functional.conv1d(
            arr_1d_torch,
            kers_1d_torch,
            stride=stride_1d,
            padding=pad_1d,
        )

        arr_1d_node = nodes.ConstantNode(arr_1d)
        kers_1d_node = nodes.ConstantNode(kers_1d)
        res_1d_node = nodes.MultichannelConvolutionNode(arr_1d_node, kers_1d_node, pad_1d, stride_1d, self.get_multichannel_convolution_version())

        output_grad_1d = self._rng.standard_normal(res_1d_node.get_shape())
        input_grad_1d, = res_1d_node.get_gradients_against([arr_1d_node], output_grad_1d)
        res_1d_torch.backward(torch.tensor(output_grad_1d, requires_grad=False))

        self.assertTrue(np.allclose(arr_1d_torch.grad.detach().numpy(), input_grad_1d))

        arr_2d = self._rng.standard_normal((100,10,50,50))
        kers_size_2d = (12,10,13,11)
        kers_2d = self._rng.standard_normal(kers_size_2d)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d, requires_grad=True)
        kers_2d_torch = torch.tensor(kers_2d)
        res_2d_torch = torch.nn.functional.conv2d(
            arr_2d_torch,
            kers_2d_torch,
            stride=stride_2d,
            padding=pad_2d,
        )

        arr_2d_node = nodes.ConstantNode(arr_2d)
        kers_2d_node = nodes.ConstantNode(kers_2d)
        res_2d_node = nodes.MultichannelConvolutionNode(arr_2d_node, kers_2d_node, pad_2d, stride_2d, self.get_multichannel_convolution_version())

        output_grad_2d = self._rng.standard_normal(res_2d_node.get_shape())
        input_grad_2d, = res_2d_node.get_gradients_against([arr_2d_node], output_grad_2d)
        res_2d_torch.backward(torch.tensor(output_grad_2d, requires_grad=False))

        self.assertTrue(np.allclose(arr_2d_torch.grad.detach().numpy(), input_grad_2d))

        arr_3d = self._rng.standard_normal((100,10,50,50,50))
        kers_size_3d = (11,10,5,4,3)
        kers_3d = self._rng.standard_normal(kers_size_3d)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d, requires_grad=True)
        kers_3d_torch = torch.tensor(kers_3d)
        res_3d_torch = torch.nn.functional.conv3d(
            arr_3d_torch,
            kers_3d_torch,
            stride=stride_3d,
            padding=pad_3d,
        )

        arr_3d_node = nodes.ConstantNode(arr_3d)
        kers_3d_node = nodes.ConstantNode(kers_3d)
        res_3d_node = nodes.MultichannelConvolutionNode(arr_3d_node, kers_3d_node, pad_3d, stride_3d, self.get_multichannel_convolution_version())

        output_grad_3d = self._rng.standard_normal(res_3d_node.get_shape())
        input_grad_3d, = res_3d_node.get_gradients_against([arr_3d_node], output_grad_3d)
        res_3d_torch.backward(torch.tensor(output_grad_3d, requires_grad=False))

        self.assertTrue(np.allclose(arr_3d_torch.grad.detach().numpy(), input_grad_3d))

    def test_multichannel_convolution_kernel_derivative(self):
        arr_1d = self._rng.standard_normal((100,10,50))
        kers_size_1d = (11,10,13,)
        kers_1d = self._rng.standard_normal(kers_size_1d)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d)
        kers_1d_torch = torch.tensor(kers_1d, requires_grad=True)
        res_1d_torch: torch.Tensor = torch.nn.functional.conv1d(
            arr_1d_torch,
            kers_1d_torch,
            stride=stride_1d,
            padding=pad_1d,
        )

        arr_1d_node = nodes.ConstantNode(arr_1d)
        kers_1d_node = nodes.ConstantNode(kers_1d)
        res_1d_node = nodes.MultichannelConvolutionNode(arr_1d_node, kers_1d_node, pad_1d, stride_1d, self.get_multichannel_convolution_version())

        output_grad_1d = self._rng.standard_normal(res_1d_node.get_shape())
        kers_grad_1d, = res_1d_node.get_gradients_against([kers_1d_node], output_grad_1d)
        res_1d_torch.backward(torch.tensor(output_grad_1d, requires_grad=False))

        self.assertTrue(np.allclose(kers_1d_torch.grad.detach().numpy(), kers_grad_1d))

        arr_2d = self._rng.standard_normal((100,10,50,50))
        kers_size_2d = (12,10,13,11)
        kers_2d = self._rng.standard_normal(kers_size_2d)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d)
        kers_2d_torch = torch.tensor(kers_2d, requires_grad=True)
        res_2d_torch = torch.nn.functional.conv2d(
            arr_2d_torch,
            kers_2d_torch,
            stride=stride_2d,
            padding=pad_2d,
        )

        arr_2d_node = nodes.ConstantNode(arr_2d)
        kers_2d_node = nodes.ConstantNode(kers_2d)
        res_2d_node = nodes.MultichannelConvolutionNode(arr_2d_node, kers_2d_node, pad_2d, stride_2d, self.get_multichannel_convolution_version())

        output_grad_2d = self._rng.standard_normal(res_2d_node.get_shape())
        kers_grad_2d, = res_2d_node.get_gradients_against([kers_2d_node], output_grad_2d)
        res_2d_torch.backward(torch.tensor(output_grad_2d, requires_grad=False))

        self.assertTrue(np.allclose(kers_2d_torch.grad.detach().numpy(), kers_grad_2d))

        arr_3d = self._rng.standard_normal((100,10,50,50,50))
        kers_size_3d = (11,10,5,4,3)
        kers_3d = self._rng.standard_normal(kers_size_3d)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d)
        kers_3d_torch = torch.tensor(kers_3d, requires_grad=True)
        res_3d_torch = torch.nn.functional.conv3d(
            arr_3d_torch,
            kers_3d_torch,
            stride=stride_3d,
            padding=pad_3d,
        )

        arr_3d_node = nodes.ConstantNode(arr_3d)
        ker_3d_node = nodes.ConstantNode(kers_3d)
        res_3d_node = nodes.MultichannelConvolutionNode(arr_3d_node, ker_3d_node, pad_3d, stride_3d, self.get_multichannel_convolution_version())

        output_grad_3d = self._rng.standard_normal(res_3d_node.get_shape())
        kers_grad_3d, = res_3d_node.get_gradients_against([ker_3d_node], output_grad_3d)
        res_3d_torch.backward(torch.tensor(output_grad_3d, requires_grad=False))

        self.assertTrue(np.allclose(kers_3d_torch.grad.detach().numpy(), kers_grad_3d))

    def test_nested_convolution(self):
        arr_2d = self._rng.standard_normal((10, 1, 100, 100))
        ker1 = self._rng.standard_normal((10, 1, 10, 10))
        ker2 = self._rng.standard_normal((10, 10, 6, 6))
        ker3 = self._rng.standard_normal((20, 10, 3, 3))
        out_grad = self._rng.standard_normal((10, 20, 19, 19))

        arr_2d_torch = torch.tensor(arr_2d, requires_grad=True)
        ker1_torch, ker2_torch, ker3_torch = [
            torch.tensor(ker, requires_grad=True)
            for ker in (ker1, ker2, ker3)
        ]

        temp1_torch = torch.nn.functional.conv2d(arr_2d_torch, ker1_torch, stride=2)
        temp2_torch = torch.nn.functional.conv2d(temp1_torch, ker2_torch, stride=2)
        temp3_torch = torch.nn.functional.conv2d(temp2_torch, ker3_torch, stride=1)

        temp3_torch.backward(torch.tensor(out_grad, requires_grad=False))

        arr_2d_node = nodes.ConstantNode(arr_2d)
        ker1_node, ker2_node, ker3_node = [
            nodes.ConstantNode(ker)
            for ker in (ker1, ker2, ker3)
        ]

        temp1_node = nodes.MultichannelConvolutionNode(arr_2d_node, ker1_node, (0, 0), (2, 2))
        temp2_node = nodes.MultichannelConvolutionNode(temp1_node, ker2_node, (0, 0), (2, 2))
        temp3_node = nodes.MultichannelConvolutionNode(temp2_node, ker3_node, (0, 0), (1, 1))

        arr_2d_grad, ker1_grad, ker2_grad, ker3_grad = temp3_node.get_gradients_against(
            [arr_2d_node, ker1_node, ker2_node, ker3_node], out_grad
        )

        self.assertTrue(np.allclose(temp3_torch.detach().numpy(), temp3_node.get_value()))
        self.assertTrue(np.allclose(ker3_torch.grad.detach().numpy(), ker3_grad))
        self.assertTrue(np.allclose(ker2_torch.grad.detach().numpy(), ker2_grad))
        self.assertTrue(np.allclose(ker1_torch.grad.detach().numpy(), ker1_grad))
        self.assertTrue(np.allclose(arr_2d_torch.grad.detach().numpy(), arr_2d_grad))

class MultichannelConvolutionNodeV1TestCase(MultichannelConvolutionNodeCommonTests, unittest.TestCase):
    @override
    def get_multichannel_convolution_version(self):
        return 1
    
class MultichannelConvolutionNodeV2TestCase(MultichannelConvolutionNodeCommonTests, unittest.TestCase):
    @override
    def get_multichannel_convolution_version(self):
        return 2
