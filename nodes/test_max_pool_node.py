
from typing import *
import unittest
import numpy as np
import torch
import nodes

class MaxPoolNodeTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(10012002)

    def test_max_pool_shape(self):
        arr_1d = np.zeros((100,10,50))
        ker_size_1d = (13,)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d)
        res_1d_torch = torch.nn.functional.max_pool1d(arr_1d_torch, ker_size_1d, stride_1d, pad_1d)

        arr_1d_node = nodes.ConstantNode(arr_1d)
        res_1d_node = nodes.MaxPoolNode(arr_1d_node, ker_size_1d, pad_1d, stride_1d)

        self.assertEqual(tuple(res_1d_torch.shape), res_1d_node.get_shape())

        arr_2d = np.zeros((100,10,50,50))
        ker_size_2d = (13,11)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d)
        res_2d_torch = torch.nn.functional.max_pool2d(arr_2d_torch, ker_size_2d, stride_2d, pad_2d)

        arr_2d_node = nodes.ConstantNode(arr_2d)
        res_2d_node = nodes.MaxPoolNode(arr_2d_node, ker_size_2d, pad_2d, stride_2d)

        self.assertEqual(tuple(res_2d_torch.shape), res_2d_node.get_shape())

        arr_3d = np.zeros((100,10,50,50,50))
        ker_size_3d = (5,4,3)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d)
        res_3d_torch = torch.nn.functional.max_pool3d(arr_3d_torch, ker_size_3d, stride_3d, pad_3d)

        arr_3d_node = nodes.ConstantNode(arr_3d)
        res_3d_node = nodes.MaxPoolNode(arr_3d_node, ker_size_3d, pad_3d, stride_3d)

        self.assertEqual(tuple(res_3d_torch.shape), res_3d_node.get_shape())

    def test_max_pool_value(self):
        arr_1d = self._rng.standard_normal((100,10,50))
        ker_size_1d = (13,)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d)
        res_1d_torch = torch.nn.functional.max_pool1d(arr_1d_torch, ker_size_1d, stride_1d, pad_1d)

        arr_1d_node = nodes.ConstantNode(arr_1d)
        res_1d_node = nodes.MaxPoolNode(arr_1d_node, ker_size_1d, pad_1d, stride_1d)

        self.assertTrue(np.allclose(res_1d_torch.detach().numpy(), res_1d_node.get_value()))

        arr_2d = self._rng.standard_normal((100,10,50,50))
        ker_size_2d = (13,11)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d)
        res_2d_torch = torch.nn.functional.max_pool2d(arr_2d_torch, ker_size_2d, stride_2d, pad_2d)

        arr_2d_node = nodes.ConstantNode(arr_2d)
        res_2d_node = nodes.MaxPoolNode(arr_2d_node, ker_size_2d, pad_2d, stride_2d)

        self.assertTrue(np.allclose(res_2d_torch.detach().numpy(), res_2d_node.get_value()))

        arr_3d = self._rng.standard_normal((100,10,50,50,50))
        ker_size_3d = (5,4,3)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d)
        res_3d_torch = torch.nn.functional.max_pool3d(arr_3d_torch, ker_size_3d, stride_3d, pad_3d)

        arr_3d_node = nodes.ConstantNode(arr_3d)
        res_3d_node = nodes.MaxPoolNode(arr_3d_node, ker_size_3d, pad_3d, stride_3d)

        self.assertTrue(np.allclose(res_3d_torch.detach().numpy(), res_3d_node.get_value()))

    def test_max_pool_derivative(self):
        arr_1d = self._rng.standard_normal((100,10,50))
        ker_size_1d = (13,)
        pad_1d = (4,)
        stride_1d = (3,)

        arr_1d_torch = torch.tensor(arr_1d, requires_grad=True)
        res_1d_torch = torch.nn.functional.max_pool1d(arr_1d_torch, ker_size_1d, stride_1d, pad_1d)

        arr_1d_node = nodes.ConstantNode(arr_1d)
        res_1d_node = nodes.MaxPoolNode(arr_1d_node, ker_size_1d, pad_1d, stride_1d)

        output_grad_1d = self._rng.standard_normal(res_1d_node.get_shape())
        input_grad_1d, = res_1d_node.get_gradients_against([arr_1d_node], output_grad_1d)
        res_1d_torch.backward(torch.tensor(output_grad_1d, requires_grad=False))

        self.assertTrue(np.allclose(arr_1d_torch.grad.detach().numpy(), input_grad_1d))

        arr_2d = self._rng.standard_normal((100,10,50,50))
        ker_size_2d = (13,11)
        pad_2d = (4,3)
        stride_2d = (3,4)

        arr_2d_torch = torch.tensor(arr_2d, requires_grad=True)
        res_2d_torch = torch.nn.functional.max_pool2d(arr_2d_torch, ker_size_2d, stride_2d, pad_2d)

        arr_2d_node = nodes.ConstantNode(arr_2d)
        res_2d_node = nodes.MaxPoolNode(arr_2d_node, ker_size_2d, pad_2d, stride_2d)

        output_grad_2d = self._rng.standard_normal(res_2d_node.get_shape())
        input_grad_2d = res_2d_node.get_gradients_against([arr_2d_node], output_grad_2d)
        res_2d_torch.backward(torch.tensor(output_grad_2d, requires_grad=False))

        self.assertTrue(np.allclose(arr_2d_torch.grad.detach().numpy(), input_grad_2d))

        arr_3d = self._rng.standard_normal((100,10,50,50,50))
        ker_size_3d = (5,4,3)
        pad_3d = (2,1,1)
        stride_3d = (3,4,5)

        arr_3d_torch = torch.tensor(arr_3d, requires_grad=True)
        res_3d_torch = torch.nn.functional.max_pool3d(arr_3d_torch, ker_size_3d, stride_3d, pad_3d)

        arr_3d_node = nodes.ConstantNode(arr_3d)
        res_3d_node = nodes.MaxPoolNode(arr_3d_node, ker_size_3d, pad_3d, stride_3d)

        output_grad_3d = self._rng.standard_normal(res_3d_node.get_shape())
        input_grad_3d = res_3d_node.get_gradients_against([arr_3d_node], output_grad_3d)
        res_3d_torch.backward(torch.tensor(output_grad_3d, requires_grad=False))

        self.assertTrue(np.allclose(arr_3d_torch.grad.detach().numpy(), input_grad_3d))

