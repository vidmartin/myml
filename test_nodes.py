
from typing import *
import itertools
import unittest
import nodes, elementwise, utils
from permutation import Permutation
import torch
import numpy as np

# TODO: split into multiple files

class NodesTestCase(unittest.TestCase):
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
        out_node = nodes.TransposeNode(A_node, Permutation.create(permutation))
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

    def test_sum(self) -> None:
        A = self._rng.random((4, 4, 4, 4, 4))
        output_grad = self._rng.random((4, 4))

        A_torch = torch.tensor(A, requires_grad=True)
        out_torch = A_torch.sum((0, 1, 2))
        out_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        out_node = nodes.SumNode(A_node, 3)
        A_grad, = out_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(out_node.get_value(), out_torch.detach().numpy()))
        
        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))

    def test_avg(self) -> None:
        A = self._rng.random((4, 4, 4, 4, 4))
        output_grad = self._rng.random((4, 4))

        A_torch = torch.tensor(A, requires_grad=True)
        out_torch = A_torch.mean((0, 1, 2))
        out_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        out_node = nodes.AvgNode(A_node, 3)
        A_grad, = out_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(out_node.get_value(), out_torch.detach().numpy()))
        
        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))

    def test_reshape(self) -> None:
        A = self._rng.random((4, 4, 4, 4, 4))
        output_grad = self._rng.random((4, 4 * 4 * 4, 4))

        A_torch = torch.tensor(A, requires_grad=True)
        out_torch = A_torch.reshape((4, -1, 4))
        out_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        out_node = nodes.ReshapeNode(A_node, (4, -1, 4))
        A_grad, = out_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(out_node.get_value(), out_torch.detach().numpy()))
        
        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))

    def test_mse(self) -> None:
        A, W, y = self._rng.random((100, 4)), self._rng.random((4, 1)), self._rng.random((100, 1))
        
        A_torch, W_torch, y_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, W, y)]
        tensordot_torch = torch.tensordot(A_torch, W_torch, 1)
        mse_torch = torch.nn.functional.mse_loss(tensordot_torch, y_torch, reduction="mean")
        mse_torch.backward()

        A_node, W_node, y_node = [nodes.ConstantNode(arr) for arr in (A, W, y)]
        tensordot_node = nodes.TensorDotNode(A_node, W_node, 1)
        mse_node = nodes.AvgNode(nodes.ElementwiseNode(elementwise.ElementwiseSquaredDifference(), [tensordot_node, y_node]), 1)
        A_grad, W_grad = mse_node.get_gradients_against([A_node, W_node])

        self.assertTrue(
            np.allclose(mse_node.get_value(), mse_torch.detach().numpy().reshape(mse_node.get_shape())),
            f"got {mse_node.get_value()}, should be {mse_torch}"
        )
        # ^ we care mainly about the value, so we ignore the shape

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(W_grad, W_torch.grad.detach().numpy()))

    def test_mse_wrapped(self) -> None:
        A, W, y = self._rng.random((100, 4)), self._rng.random((4, 1)), self._rng.random((100, 1))
        
        A_torch, W_torch, y_torch = [torch.tensor(arr, requires_grad=True) for arr in (A, W, y)]
        tensordot_torch = torch.tensordot(A_torch, W_torch, 1)
        mse_torch = torch.nn.functional.mse_loss(tensordot_torch, y_torch, reduction="mean")
        mse_torch.backward()

        A_node, W_node, y_node = [nodes.ConstantNode(arr) for arr in (A, W, y)]
        tensordot_node = nodes.TensorDotNode(A_node, W_node, 1)
        wrapped_mse_node = nodes.WrappingNode(
            [tensordot_node],
            lambda nodes_: nodes.AvgNode(
                nodes.ElementwiseNode(elementwise.ElementwiseSquaredDifference(), [nodes_[0], y_node]), 1
            )
        )
        A_grad, W_grad = wrapped_mse_node.get_gradients_against([A_node, W_node])
        self.assertTrue(
            np.allclose(wrapped_mse_node.get_value(), mse_torch.detach().numpy().reshape(wrapped_mse_node.get_shape())),
            f"got {wrapped_mse_node.get_value()}, should be {mse_torch}"
        )
        # ^ we care mainly about the value, so we ignore the shape

        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()))
        self.assertTrue(np.allclose(W_grad, W_torch.grad.detach().numpy()))

    def test_logsumexp(self) -> None:
        output_grad = self._rng.random((20,))
        for k in (1, 10, 100, 1000):
            X = self._rng.random((20, 20)) * k

            X_torch = torch.tensor(X, requires_grad=True)
            logsumexp_torch = torch.logsumexp(X_torch, -1)
            logsumexp_torch.backward(torch.tensor(output_grad, requires_grad=False))

            X_node = nodes.ConstantNode(X)
            logsumexp_node = nodes.LogSumExpNode(X_node)
            grad, = logsumexp_node.get_gradients_against([X_node], output_grad)

            self.assertTrue(np.allclose(logsumexp_node.get_value(), logsumexp_torch.detach().numpy(), atol=0.001), f"value doesn't match, k = {k}")

            self.assertTrue(np.allclose(grad, X_torch.grad.detach().numpy(), atol=0.001), f"grad doesn't match, k = {k}, expected {X_torch.grad.detach().numpy()}, but got {grad}")
    
    def test_softmax(self) -> None:
        A = self._rng.random((20,8)) * 100
        output_grad = self._rng.random(A.shape)

        A_torch = torch.tensor(A, requires_grad=True)
        softmax_torch = torch.softmax(A_torch, -1)
        softmax_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        softmax_node = nodes.SoftmaxNode(A_node)
        A_grad, = softmax_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(A_node.get_value(), A_torch.detach().numpy()))
        
        self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy()), f"grad doesn't match, expected {A_torch.grad.detach().numpy()}, but got {A_grad}")

    def test_cross_entropy_logits(self) -> None:
        yhat, y = [self._rng.random((20, 8)) for _ in range(2)]
        yhat *= 100
        y = y / y.sum(-1)[...,np.newaxis]
        output_grad = self._rng.random((20,))

        yhat_torch, y_torch = [torch.tensor(arr, requires_grad=True) for arr in (yhat, y)]
        cross_entropy_torch = torch.nn.functional.cross_entropy(yhat_torch, y_torch, reduction="none")
        cross_entropy_torch.backward(torch.tensor(output_grad))

        yhat_node, y_node = [nodes.ConstantNode(arr) for arr in (yhat, y)]
        cross_entropy_node = nodes.CrossEntropyLogitsNode(yhat_node, y_node)
        yhat_grad, y_grad = cross_entropy_node.get_gradients_against([yhat_node, y_node], output_grad)

        self.assertTrue(np.allclose(cross_entropy_node.get_value(), cross_entropy_torch.detach().numpy(), atol=0.001))

        self.assertTrue(np.allclose(yhat_grad, yhat_torch.grad.detach().numpy(), atol=0.001))
        self.assertTrue(np.allclose(y_grad, y_torch.grad.detach().numpy(), atol=0.001), f"expected\n{y_torch.grad}\n but got \n{y_grad}")

    def test_fully_connected_regression(self) -> None:
        X, y = self._rng.random((100, 25)), self._rng.random((100, 1))
        weights = [
            self._rng.random((25, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 1))
        ]
        biases = [
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((1,))
        ]
        
        X_torch = torch.tensor(X, requires_grad=False)
        weights_torch = [
            torch.tensor(arr, requires_grad=True)
            for arr in weights
        ]
        biases_torch = [
            torch.tensor(arr, requires_grad=True)
            for arr in biases
        ]
        
        temp_torch = X_torch
        for W, b in zip(weights_torch, biases_torch):
            temp_torch = torch.nn.functional.relu(temp_torch @ W + b)
        loss_torch = torch.nn.functional.mse_loss(temp_torch, torch.tensor(y), reduction="mean")
        loss_torch.backward()

        X_node, y_node = nodes.ConstantNode(X), nodes.ConstantNode(y)
        weights_nodes = [
            nodes.ConstantNode(arr)
            for arr in weights
        ]
        biases_nodes = [
            nodes.ConstantNode(arr)
            for arr in biases
        ]

        temp_node = X_node
        for W, b in zip(weights_nodes, biases_nodes):
            temp_node = nodes.ElementwiseNode(
                elementwise.ElementwiseReLU(), [
                    nodes.ElementwiseNode(
                        elementwise.ElementwiseAdd(2), [
                            nodes.TensorDotNode(temp_node, W, 1),
                            nodes.ExtendNode(b, (X.shape[0],))
                        ]
                    )
                ]
            )
        loss_node = nodes.AvgNode(nodes.ElementwiseNode(elementwise.ElementwiseSquaredDifference(), [temp_node, y_node]), 1)
        grads = loss_node.get_gradients_against(weights_nodes + biases_nodes)

        self.assertTrue(np.allclose(loss_node.get_value(), loss_torch.detach().numpy()))

        for torch_tensor, my_grad in zip(weights_torch + biases_torch, grads):
            self.assertTrue(np.allclose(my_grad, torch_tensor.grad.detach().numpy()))

    def test_fully_connected_regression_tied(self) -> None:
        X, y = self._rng.random((100, 25)), self._rng.random((100, 1))
        weights = [
            self._rng.random((25, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 10)),
            1,
            self._rng.random((10, 1))
        ]
        biases = [
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((10,)),
            1,
            self._rng.random((1,))
        ]
        
        X_torch = torch.tensor(X, requires_grad=False)
        weights_torch = [
            torch.tensor(arr, requires_grad=True)
            if not isinstance(arr, int) else None
            for arr in weights
        ]
        biases_torch = [
            torch.tensor(arr, requires_grad=True)
            if not isinstance(arr, int) else None
            for arr in biases
        ]

        for i, W in enumerate(weights):
            if isinstance(W, int):
                weights_torch[i] = weights_torch[W]
        for i, W in enumerate(biases):
            if isinstance(W, int):
                biases_torch[i] = biases_torch[W]
        
        temp_torch = X_torch
        for W, b in zip(weights_torch, biases_torch):
            temp_torch = torch.nn.functional.relu(temp_torch @ W + b)
        loss_torch = torch.nn.functional.mse_loss(temp_torch, torch.tensor(y), reduction="mean")
        loss_torch.backward()

        X_node, y_node = nodes.ConstantNode(X), nodes.ConstantNode(y)
        weights_nodes = [
            nodes.ConstantNode(arr)
            if not isinstance(arr, int) else None
            for arr in weights
        ]
        biases_nodes = [
            nodes.ConstantNode(arr)
            if not isinstance(arr, int) else None
            for arr in biases
        ]

        for i, W in enumerate(weights):
            if isinstance(W, int):
                weights_nodes[i] = weights_nodes[W]
        for i, W in enumerate(biases):
            if isinstance(W, int):
                biases_nodes[i] = biases_nodes[W]

        temp_node = X_node
        for W, b in zip(weights_nodes, biases_nodes):
            temp_node = nodes.ElementwiseNode(
                elementwise.ElementwiseReLU(), [
                    nodes.ElementwiseNode(
                        elementwise.ElementwiseAdd(2), [
                            nodes.TensorDotNode(temp_node, W, 1),
                            nodes.ExtendNode(b, (X.shape[0],))
                        ]
                    )
                ]
            )
        loss_node = nodes.AvgNode(nodes.ElementwiseNode(elementwise.ElementwiseSquaredDifference(), [temp_node, y_node]), 1)
        grads = loss_node.get_gradients_against(weights_nodes + biases_nodes)

        self.assertTrue(np.allclose(loss_node.get_value(), loss_torch.detach().numpy()))

        for torch_tensor, my_grad in zip(weights_torch + biases_torch, grads):
            self.assertTrue(np.allclose(my_grad, torch_tensor.grad.detach().numpy()))

    def test_fully_connected_classification(self) -> None:
        X, y = self._rng.random((100, 25)), self._rng.integers(0, 5, (100,))
        weights = [
            self._rng.random((25, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 5))
        ]
        biases = [
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((5,))
        ]
        
        X_torch = torch.tensor(X, requires_grad=False)
        weights_torch = [
            torch.tensor(arr, requires_grad=True)
            for arr in weights
        ]
        biases_torch = [
            torch.tensor(arr, requires_grad=True)
            for arr in biases
        ]
        
        temp_torch = X_torch
        for W, b in zip(weights_torch, biases_torch):
            temp_torch = torch.nn.functional.relu(temp_torch @ W + b)
        loss_torch = torch.nn.functional.cross_entropy(temp_torch, torch.tensor(y), reduction="mean")
        loss_torch.backward()

        X_node = nodes.ConstantNode(X)
        weights_nodes = [
            nodes.ConstantNode(arr)
            for arr in weights
        ]
        biases_nodes = [
            nodes.ConstantNode(arr)
            for arr in biases
        ]

        temp_node = X_node
        for W, b in zip(weights_nodes, biases_nodes):
            temp_node = nodes.ElementwiseNode(
                elementwise.ElementwiseReLU(), [
                    nodes.ElementwiseNode(
                        elementwise.ElementwiseAdd(2), [
                            nodes.TensorDotNode(temp_node, W, 1),
                            nodes.ExtendNode(b, (X.shape[0],))
                        ]
                    )
                ]
            )
        target_probas_node = nodes.ConstantNode(utils.one_hot_encode(y, 5))
        loss_node = nodes.AvgNode(nodes.CrossEntropyLogitsNode(temp_node, target_probas_node), 1)
        grads = loss_node.get_gradients_against(weights_nodes + biases_nodes)

        self.assertTrue(np.allclose(loss_node.get_value(), loss_torch.detach().numpy()))

        for torch_tensor, my_grad in zip(weights_torch + biases_torch, grads):
            self.assertTrue(np.allclose(my_grad, torch_tensor.grad.detach().numpy()))

    def test_fully_connected_classification_tied(self) -> None:
        X, y = self._rng.random((100, 25)), self._rng.integers(0, 5, (100,))
        weights = [
            self._rng.random((25, 10)),
            self._rng.random((10, 10)),
            self._rng.random((10, 10)),
            1,
            self._rng.random((10, 5))
        ]
        biases = [
            self._rng.random((10,)),
            self._rng.random((10,)),
            self._rng.random((10,)),
            1,
            self._rng.random((5,))
        ]
        
        X_torch = torch.tensor(X, requires_grad=False)
        weights_torch = [
            torch.tensor(arr, requires_grad=True)
            if not isinstance(arr, int) else None
            for arr in weights
        ]
        biases_torch = [
            torch.tensor(arr, requires_grad=True)
            if not isinstance(arr, int) else None
            for arr in biases
        ]

        for i, W in enumerate(weights):
            if isinstance(W, int):
                weights_torch[i] = weights_torch[W]
        for i, W in enumerate(biases):
            if isinstance(W, int):
                biases_torch[i] = biases_torch[W]
        
        temp_torch = X_torch
        for W, b in zip(weights_torch, biases_torch):
            temp_torch = torch.nn.functional.relu(temp_torch @ W + b)
        loss_torch = torch.nn.functional.cross_entropy(temp_torch, torch.tensor(y), reduction="mean")
        loss_torch.backward()

        X_node = nodes.ConstantNode(X)
        weights_nodes = [
            nodes.ConstantNode(arr)
            if not isinstance(arr, int) else None
            for arr in weights
        ]
        biases_nodes = [
            nodes.ConstantNode(arr)
            if not isinstance(arr, int) else None
            for arr in biases
        ]

        for i, W in enumerate(weights):
            if isinstance(W, int):
                weights_nodes[i] = weights_nodes[W]
        for i, W in enumerate(biases):
            if isinstance(W, int):
                biases_nodes[i] = biases_nodes[W]

        temp_node = X_node
        for W, b in zip(weights_nodes, biases_nodes):
            temp_node = nodes.ElementwiseNode(
                elementwise.ElementwiseReLU(), [
                    nodes.ElementwiseNode(
                        elementwise.ElementwiseAdd(2), [
                            nodes.TensorDotNode(temp_node, W, 1),
                            nodes.ExtendNode(b, (X.shape[0],))
                        ]
                    )
                ]
            )
        target_probas_node = nodes.ConstantNode(utils.one_hot_encode(y, 5))
        loss_node = nodes.AvgNode(nodes.CrossEntropyLogitsNode(temp_node, target_probas_node), 1)
        grads = loss_node.get_gradients_against(weights_nodes + biases_nodes)

        self.assertTrue(np.allclose(loss_node.get_value(), loss_torch.detach().numpy()))

        for torch_tensor, my_grad in zip(weights_torch + biases_torch, grads):
            self.assertTrue(np.allclose(my_grad, torch_tensor.grad.detach().numpy()))
