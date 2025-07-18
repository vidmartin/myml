
from typing import *
import itertools
import unittest
import nodes
import elementwise
import torch
import numpy as np

class TestNodes(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(10012002)

    def test_add(self) -> None:
        A, B, C = [self._rng.random((4, 4)) for _ in range(3)]
        output_grad = self._rng.random((4, 4))

        A_torch, B_torch, C_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, B, C)]
        sum_torch = A_torch + B_torch + C_torch
        sum_torch.backward(torch.tensor(output_grad))

        A_node, B_node, C_node = [nodes.ConstantNode(arr) for arr in (A, B, C)]
        sum_node = nodes.ElementwiseNode(elementwise.ElementwiseAdd(3), [A_node, B_node, C_node])
        A_grad, B_grad, C_grad = sum_node.get_gradients_against([A_node, B_node, C_node], output_grad)

        self.assertTrue(np.allclose(sum_node.get_value(), sum_torch.detach().numpy()))

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(B_grad, B_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(C_grad, C_torch.grad.detach().numpy()))

    def test_mul(self) -> None:
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

    def test_mul_add_mul_add(self) -> None:
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

        self.assertTrue(np.allclose(add2_node.get_value(), add2_torch.detach().numpy()))

        for grad, ref_tensor in zip(grads, arrs_torch):
            self.assertTrue(np.allclose(grad, ref_tensor.grad.detach().numpy()))

    def test_tensordot(self) -> None:
        A, B = [self._rng.random((4, 4, 4)) for _ in range(2)]
        output_grad = self._rng.random((4, 4))

        A_torch, B_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, B)]
        out_torch: torch.Tensor = torch.tensordot(A_torch, B_torch, 2)
        out_torch.backward(torch.tensor(output_grad))

        A_node, B_node = [nodes.ConstantNode(arr) for arr in (A, B)]
        out_node = nodes.TensorDotNode(A_node, B_node, 2)
        A_grad, B_grad = out_node.get_gradients_against([A_node, B_node], output_grad)

        self.assertTrue(np.allclose(out_node.get_value(), out_torch.detach().numpy()))

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(B_grad, B_torch.grad.detach().numpy()))

    def test_transpose(self) -> None:
        A = self._rng.random((4,) * 6)
        output_grad = self._rng.random(A.shape)

        permutation = list(range(len(A.shape)))
        self._rng.shuffle(permutation)
        permutation = tuple(permutation)

        A_torch = torch.tensor(A, requires_grad=True)
        out_torch = A_torch.permute(permutation)
        out_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        out_node = nodes.TransposeNode(A_node, permutation)
        A_grad, = out_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(out_node.get_value(), out_torch.detach().numpy()))

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))

    def test_extend(self) -> None:
        A = self._rng.random((4, 4))
        output_grad = self._rng.random((3, 3, 3, 4, 4))

        A_torch = torch.tensor(A, requires_grad=True)
        out_torch = A_torch[torch.newaxis,torch.newaxis,torch.newaxis,:,:] + torch.zeros((3, 3, 3, 1, 1))
        out_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        out_node = nodes.ExtendNode(A_node, (3, 3, 3))
        A_grad, = out_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(out_node.get_value(), out_torch.detach().numpy()))

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))

    def test_mse(self) -> None:
        A, W, y = self._rng.random((10, 4)), self._rng.random((4, 1)), self._rng.random((10, 1))
        
        A_torch, W_torch, y_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, W, y)]
        tensordot_torch = torch.tensordot(A_torch, W_torch, 1)
        mse_torch = torch.nn.functional.mse_loss(tensordot_torch, y_torch, reduction="mean")
        mse_torch.backward()

        A_node, W_node, y_node = [nodes.ConstantNode(arr) for arr in (A, W, y)]
        tensordot_node = nodes.TensorDotNode(A_node, W_node, 1)
        mse_node = nodes.mse_node(tensordot_node, y)
        A_grad, W_grad = mse_node.get_gradients_against([A_node, W_node])

        self.assertTrue(
            np.allclose(mse_node.get_value(), mse_torch.detach().numpy().reshape(mse_node.get_shape())),
            f"got {mse_node.get_value()}, should be {mse_torch}"
        )
        # ^ we care mainly about the value, so we ignore the shape

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(W_grad, W_torch.grad.detach().numpy()))
