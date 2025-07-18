
from typing import *
import itertools
import unittest
import nodes
import elementwise
import torch
import numpy as np

class TestGradientComputation(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(10012002)

    def test_add(self):
        A, B, C = [self._rng.random((4, 4)) for _ in range(3)]
        output_grad = self._rng.random((4, 4))

        A_torch, B_torch, C_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, B, C)]
        sum_torch = A_torch + B_torch + C_torch
        sum_torch.backward(torch.tensor(output_grad))

        A_node, B_node, C_node = [nodes.ConstantNode(arr) for arr in (A, B, C)]
        sum_node = nodes.ElementwiseNode(elementwise.ElementwiseAdd(3), [A_node, B_node, C_node])
        A_grad, B_grad, C_grad = sum_node.get_gradients_against([A_node, B_node, C_node], output_grad)

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(B_grad, B_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(C_grad, C_torch.grad.detach().numpy()))

    def test_mul(self):
        A, B, C = [self._rng.random((4, 4)) for _ in range(3)]
        output_grad = self._rng.random((4, 4))

        A_torch, B_torch, C_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, B, C)]
        sum_torch = A_torch * B_torch * C_torch
        sum_torch.backward(torch.tensor(output_grad))

        A_node, B_node, C_node = [nodes.ConstantNode(arr) for arr in (A, B, C)]
        sum_node = nodes.ElementwiseNode(elementwise.ElementwiseMul(3), [A_node, B_node, C_node])
        A_grad, B_grad, C_grad = sum_node.get_gradients_against([A_node, B_node, C_node], output_grad)

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(B_grad, B_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(C_grad, C_torch.grad.detach().numpy()))

    def test_mul_add_mul_add(self):
        arrs = [self._rng.random((4, 4)) for _ in range(16)]
        output_grad = self._rng.random((4, 4))

        arrs_torch = [torch.tensor(arr, requires_grad=True) for arr in arrs]
        mul1_torch = [a * b for a, b in itertools.batched(arrs_torch, 2)]
        add1_torch = [a + b for a, b in itertools.batched(mul1_torch, 2)]
        mul2_torch = [a * b for a, b in itertools.batched(add1_torch, 2)]
        add2_torch, = [a + b for a, b in itertools.batched(mul2_torch, 2)]
        add2_torch.backward(torch.tensor(output_grad))

        arrs_node = [nodes.ConstantNode(arr) for arr in arrs]
        mul1_node = [nodes.ElementwiseNode(elementwise.ElementwiseMul(2), [a, b]) for a, b in itertools.batched(arrs_node, 2)]
        add1_node = [nodes.ElementwiseNode(elementwise.ElementwiseAdd(2), [a, b]) for a, b in itertools.batched(mul1_node, 2)]
        mul2_node = [nodes.ElementwiseNode(elementwise.ElementwiseMul(2), [a, b]) for a, b in itertools.batched(add1_node, 2)]
        add2_node, = [nodes.ElementwiseNode(elementwise.ElementwiseAdd(2), [a, b]) for a, b in itertools.batched(mul2_node, 2)]
        grads = add2_node.get_gradients_against(arrs_node, output_grad)

        for grad, ref_tensor in zip(grads, arrs_torch):
            self.assertTrue(np.allclose(grad, ref_tensor.grad.detach().numpy()))
