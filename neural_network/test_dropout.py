
from typing import *
import unittest
import numpy as np
import torch
from neural_network.neural_network import EvaluationMode
import nodes
import utils
from neural_network.dropout import DropoutModule

TEST_SHAPES = [(100,100), (100,50,50), (100,10,10,10)]
TEST_PROBAS = np.linspace(0.1, 0.9, 9)

class DropoutTestCase(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.default_rng(101202)

    def test_dropout_training_copy_pytorch_mask(self):
        for shape in TEST_SHAPES:
            for proba in TEST_PROBAS:
                A = (self._rng.integers(0, 1, shape) * 2 - 1) * (self._rng.random(shape) + 1.0)
                # ^ we don't want zeros so that we can detect the mask that torch's dropout used
                output_grad = self._rng.random(shape)

                A_torch = torch.tensor(A, requires_grad=True)
                dropout_torch = torch.nn.Dropout(p=proba)
                dropout_torch.train()
                output_torch: torch.Tensor = dropout_torch(A_torch)
                output_torch.backward(torch.tensor(output_grad))

                torch_mask = 1.0 - (output_torch == 0.0).numpy().astype(np.float32)

                class StubRandomGenerator(utils.RandomGenerator):
                    @override
                    def __call__(self, shape: tuple[int, ...]):
                        return torch_mask

                A_node = nodes.ConstantNode(A)
                dropout_my = DropoutModule(proba, StubRandomGenerator())
                # ^ proba has no effect except the scaling, since StubRandomGenerator returns just 0s and 1s
                graph = dropout_my.construct(A_node, {}, mode=EvaluationMode.TRAINING)
                A_grad, = graph.output_node.get_gradients_against([A_node], output_grad)

                self.assertTrue(np.allclose(output_torch.detach().numpy(), graph.output_node.get_value()), f"{output_torch.detach().numpy()} vs {graph.output_node.get_value()}")
                self.assertTrue(np.allclose(A_torch.grad.detach().numpy(), A_grad))
    