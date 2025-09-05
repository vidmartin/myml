
from typing import *
import unittest
import numpy as np
import torch
import nodes
import utils

class BatchNormNodeTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(3124123)

    def test_batch_norm_node(self):
        for _ in range(5):
            A = self._rng.standard_normal((5, 5, 5, 5))
            output_grad = self._rng.standard_normal((5, 5, 5, 5))
            delta = 1e-5

            A_torch = torch.tensor(A, requires_grad=True)
            mu_ext_torch = A_torch.mean((0, 1))[torch.newaxis,torch.newaxis,...]
            sigsq_adj_ext_torch = ((A_torch - mu_ext_torch) ** 2).mean((0, 1))[torch.newaxis,torch.newaxis,...] + delta
            B_torch = (A_torch - mu_ext_torch) / sigsq_adj_ext_torch ** 0.5
            B_torch.backward(torch.tensor(output_grad, requires_grad=False))

            A_node = nodes.ConstantNode(A)
            B_node = nodes.BatchNormNode(A_node, 2, delta)
            A_grad, = B_node.get_gradients_against([A_node], output_grad)

            self.assertTrue(np.allclose(B_node.get_value(), B_torch.detach().numpy()))
            self.assertTrue(np.allclose(A_grad, A_torch.grad.detach().numpy(), atol=0.001))
