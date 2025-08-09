
from typing import *
import unittest
import numpy as np
import torch
from neural_network.neural_network import EvaluationMode
from neural_network.batch_norm import BatchNormModule
import nodes

class BatchNormTestCase(unittest.TestCase):
    @override
    def setUp(self):
        self._rng = np.random.default_rng(12344323)

    def test_batch_norm_1d(self):
        X = self._rng.standard_normal((64,10,20), dtype=np.float32)
        init_beta = self._rng.standard_normal((10,), dtype=np.float32)
        init_gamma = self._rng.standard_normal((10,), dtype=np.float32)

        X_torch = torch.tensor(X)
        batch_norm_torch_module = torch.nn.BatchNorm1d(10)
        batch_norm_torch_module.train()
        batch_norm_torch_module.get_parameter("weight").data = torch.tensor(init_gamma)
        batch_norm_torch_module.get_parameter("bias").data = torch.tensor(init_beta)
        y_torch: torch.Tensor = batch_norm_torch_module(X_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()

        X_node = nodes.ConstantNode(X)
        batch_norm_my_module = BatchNormModule(10, 1)
        graph = batch_norm_my_module.construct(
            input=X_node,
            params={
                "beta": init_beta,
                "gamma": init_gamma
            },
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        beta_grad, gamma_grad = loss_node.get_gradients_against([
            graph.param_nodes["beta"],
            graph.param_nodes["gamma"],
        ])

        self.assertTrue(np.allclose(
            y_torch.detach().numpy(),
            graph.output_node.get_value(),
            atol=0.001
        ))
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("bias").grad.detach().numpy(),
            beta_grad,
            atol=0.001
        ))
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("weight").grad.detach().numpy(),
            gamma_grad,
            atol=0.001
        ))
    def test_batch_norm_2d(self):
        X = self._rng.standard_normal((64,10,20,20), dtype=np.float32)
        init_beta = self._rng.standard_normal((10,), dtype=np.float32)
        init_gamma = self._rng.standard_normal((10,), dtype=np.float32)

        X_torch = torch.tensor(X)
        batch_norm_torch_module = torch.nn.BatchNorm2d(10)
        batch_norm_torch_module.train()
        batch_norm_torch_module.get_parameter("weight").data = torch.tensor(init_gamma)
        batch_norm_torch_module.get_parameter("bias").data = torch.tensor(init_beta)
        y_torch: torch.Tensor = batch_norm_torch_module(X_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()

        X_node = nodes.ConstantNode(X)
        batch_norm_my_module = BatchNormModule(10, 2)
        graph = batch_norm_my_module.construct(
            input=X_node,
            params={
                "beta": init_beta,
                "gamma": init_gamma
            },
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        beta_grad, gamma_grad = loss_node.get_gradients_against([
            graph.param_nodes["beta"],
            graph.param_nodes["gamma"],
        ])

        self.assertTrue(np.allclose(
            y_torch.detach().numpy(),
            graph.output_node.get_value(),
            atol=0.001
        ))
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("bias").grad.detach().numpy(),
            beta_grad,
            atol=0.001
        ))
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("weight").grad.detach().numpy(),
            gamma_grad,
            atol=0.001
        ))
    def test_batch_norm_3d(self):
        X = self._rng.standard_normal((64,10,20,20,20), dtype=np.float32)
        init_beta = self._rng.standard_normal((10,), dtype=np.float32)
        init_gamma = self._rng.standard_normal((10,), dtype=np.float32)

        X_torch = torch.tensor(X)
        batch_norm_torch_module = torch.nn.BatchNorm3d(10)
        batch_norm_torch_module.train()
        batch_norm_torch_module.get_parameter("weight").data = torch.tensor(init_gamma)
        batch_norm_torch_module.get_parameter("bias").data = torch.tensor(init_beta)
        y_torch: torch.Tensor = batch_norm_torch_module(X_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()

        X_node = nodes.ConstantNode(X)
        batch_norm_my_module = BatchNormModule(10, 3)
        graph = batch_norm_my_module.construct(
            input=X_node,
            params={
                "beta": init_beta,
                "gamma": init_gamma
            },
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        beta_grad, gamma_grad = loss_node.get_gradients_against([
            graph.param_nodes["beta"],
            graph.param_nodes["gamma"],
        ])

        self.assertTrue(np.allclose(
            y_torch.detach().numpy(),
            graph.output_node.get_value(),
            atol=0.01
        ))
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("bias").grad.detach().numpy(),
            beta_grad,
            atol=0.01
        ))
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("weight").grad.detach().numpy(),
            gamma_grad,
            atol=0.01
        ))
