
from typing import *
import unittest
import numpy as np
import torch
from neural_network.neural_network import EvaluationMode
from neural_network.batch_norm import BatchNormModule, BETA_PARAM_NAME, GAMMA_PARAM_NAME
import nodes
import utils

class BatchNormTestCase(unittest.TestCase):
    @override
    def setUp(self):
        self._rng = np.random.default_rng(12344323)

    def test_batch_norm_1d_no_affine(self):
        X = self._rng.standard_normal((64,10,20), dtype=np.float32)

        X_torch = torch.tensor(X, requires_grad=True)
        batch_norm_torch_module = torch.nn.BatchNorm1d(10, affine=False)
        batch_norm_torch_module.train()
        y_torch: torch.Tensor = batch_norm_torch_module(X_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()

        X_node = nodes.ConstantNode(X)
        batch_norm_my_module = BatchNormModule(10, 1, learnable_affine=False)
        graph = batch_norm_my_module.construct(
            input=X_node,
            params={},
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        X_grad, = loss_node.get_gradients_against([ X_node ])

        self.assertTrue(np.allclose(
            y_torch.detach().numpy(),
            graph.output_node.get_value(),
            atol=0.001
        ), f"{utils.preview_array(y_torch.detach().numpy())} vs. {utils.preview_array(graph.output_node.get_value())}")
        self.assertTrue(np.allclose(
            X_torch.grad.detach().numpy(),
            X_grad,
            atol=0.001
        ), f"{utils.preview_array(X_torch.grad.detach().numpy())} vs. {utils.preview_array(X_grad)}")

    def test_batch_norm_1d(self):
        X = self._rng.standard_normal((64,10,20), dtype=np.float32)
        init_beta = self._rng.standard_normal((10,), dtype=np.float32)
        init_gamma = self._rng.standard_normal((10,), dtype=np.float32)

        X_torch = torch.tensor(X, requires_grad=True)
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
                BETA_PARAM_NAME: init_beta,
                GAMMA_PARAM_NAME: init_gamma
            },
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        beta_grad, gamma_grad, X_grad = loss_node.get_gradients_against([
            graph.param_nodes[BETA_PARAM_NAME],
            graph.param_nodes[GAMMA_PARAM_NAME],
            X_node,
        ])

        self.assertTrue(np.allclose(
            y_torch.detach().numpy(),
            graph.output_node.get_value(),
            atol=0.001
        ), f"{utils.preview_array(y_torch.detach().numpy())} vs. {utils.preview_array(graph.output_node.get_value())}")
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("bias").grad.detach().numpy(),
            beta_grad,
            atol=0.001
        ), f"{utils.preview_array(batch_norm_torch_module.get_parameter("bias").grad.detach().numpy())} vs. {utils.preview_array(beta_grad)}")
        self.assertTrue(np.allclose(
            batch_norm_torch_module.get_parameter("weight").grad.detach().numpy(),
            gamma_grad,
            atol=0.001
        ), f"{utils.preview_array(batch_norm_torch_module.get_parameter("weight").grad.detach().numpy())} vs. {utils.preview_array(gamma_grad)}")
        self.assertTrue(np.allclose(
            X_torch.grad.detach().numpy(),
            X_grad,
            atol=0.001
        ), f"{utils.preview_array(X_torch.grad.detach().numpy())} vs. {utils.preview_array(X_grad)}")
    def test_batch_norm_2d(self):
        X = self._rng.standard_normal((64,10,20,20), dtype=np.float32)
        init_beta = self._rng.standard_normal((10,), dtype=np.float32)
        init_gamma = self._rng.standard_normal((10,), dtype=np.float32)

        X_torch = torch.tensor(X, requires_grad=True)
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
                BETA_PARAM_NAME: init_beta,
                GAMMA_PARAM_NAME: init_gamma
            },
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        beta_grad, gamma_grad, X_grad = loss_node.get_gradients_against([
            graph.param_nodes[BETA_PARAM_NAME],
            graph.param_nodes[GAMMA_PARAM_NAME],
            X_node
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
        self.assertTrue(np.allclose(
            X_torch.grad.detach().numpy(),
            X_grad,
            atol=0.001
        ))
    def test_batch_norm_3d(self):
        X = self._rng.standard_normal((64,10,20,20,20), dtype=np.float32)
        init_beta = self._rng.standard_normal((10,), dtype=np.float32)
        init_gamma = self._rng.standard_normal((10,), dtype=np.float32)

        X_torch = torch.tensor(X, requires_grad=True)
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
                BETA_PARAM_NAME: init_beta,
                GAMMA_PARAM_NAME: init_gamma
            },
            mode=EvaluationMode.TRAINING,
        )
        loss_node = nodes.SumNode(graph.output_node, len(graph.output_node.get_shape()))
        beta_grad, gamma_grad, X_grad = loss_node.get_gradients_against([
            graph.param_nodes[BETA_PARAM_NAME],
            graph.param_nodes[GAMMA_PARAM_NAME],
            X_node
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
        self.assertTrue(np.allclose(
            X_torch.grad.detach().numpy(),
            X_grad,
            atol=0.01
        ))
