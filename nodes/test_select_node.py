
from typing import *
import unittest
import numpy as np
import torch
import nodes

class SelectNodeTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(121231234)

    def test_select_node(self):
        A = self._rng.standard_normal((10, 10, 10, 10))
        output_grad = self._rng.standard_normal((10, 10, 10))
        indices = self._rng.integers(0, 10, 10)

        A_torch = torch.tensor(A, requires_grad=True)
        res_torch = A_torch[:,torch.arange(10),torch.tensor(indices),:]
        res_torch.backward(torch.tensor(output_grad))

        A_node = nodes.ConstantNode(A)
        res_node = nodes.SelectNode(A_node, 1, 2, indices)
        # A_grad, = res_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(res_torch.detach().numpy(), res_node.get_value()))
        # self.assertTrue(np.allclose(A_torch.grad.detach().numpy(), A_grad))
