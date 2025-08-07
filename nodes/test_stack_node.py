
from typing import *
import unittest
import numpy as np
import torch
import nodes

class StackNodeTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(432141)

    def test_stack_node(self):
        As = [ self._rng.standard_normal((10, 20, 40)) for _ in range(15) ]
        output_grad = self._rng.standard_normal((10, 20, 15, 40))

        As_torch = [ torch.tensor(A, requires_grad=True) for A in As ]
        res_torch = torch.stack(As_torch, dim=2)
        res_torch.backward(torch.tensor(output_grad, requires_grad=False))

        As_nodes = [ nodes.ConstantNode(A) for A in As ]
        res_node = nodes.StackNode(As_nodes, 2)
        As_grads = res_node.get_gradients_against(As_nodes, output_grad)

        self.assertTrue(np.allclose(res_torch.detach().numpy(), res_node.get_value()))

        for A_torch, A_grad in zip(As_torch, As_grads):
            self.assertTrue(np.allclose(A_torch.grad.detach().numpy(), A_grad))
