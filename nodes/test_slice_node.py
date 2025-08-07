
from typing import *
import unittest
import numpy as np
import torch
import nodes

class SliceNodeTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(1243423)

    def test_slice_node(self):
        A = self._rng.standard_normal((10, 20, 30, 40))
        output_grad = self._rng.standard_normal((10, 20, 40))
        slice_index = 7

        A_torch = torch.tensor(A, requires_grad=True)
        res_torch = A_torch[:,:,slice_index,:]
        res_torch.backward(torch.tensor(output_grad, requires_grad=False))

        A_node = nodes.ConstantNode(A)
        res_node = nodes.SliceNode(A_node, 2, slice_index)
        A_grad = res_node.get_gradients_against([A_node], output_grad)

        self.assertTrue(np.allclose(res_torch.detach().numpy(), res_node.get_value()))
        self.assertTrue(np.allclose(A_torch.grad.detach().numpy(), A_grad))
