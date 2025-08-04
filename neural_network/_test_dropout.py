
from typing import *
import numpy as np
from neural_network.dropout import DropoutModule
import utils
import nodes
import torch
import unittest

class DropoutTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(10012002)

    def test_dropout(self):
        for p in np.linspace(0, 1, 21):
            input_numpy = self._rng.random((100, 100))
            input_torch = torch.tensor(input_numpy)

            seed = self._rng.integers(0, 1000000, ()).item()

            model_torch = torch.nn.Dropout(p)
            torch.manual_seed(seed)
            result_torch = model_torch(input_torch)

            torch.manual_seed(seed)
            model_my = DropoutModule(p, utils.TorchUniformRandomGenerator())
            input_node = nodes.ConstantNode(input_numpy)
            graph = model_my.construct(input_node, {})
            result_my = graph.output_node.get_value()

            self.assertTrue(np.allclose(result_torch.numpy(), result_my), f"p = {p}, {result_torch} vs. {result_my}")
