
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
import nodes
import elementwise
import permutation

class BatchNormModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(
        self,
        n_features: int,
        spatial_dims: int,
        rho: float = 0.1,
        delta: float = 1e-5,
        learnable_affine: bool = True,
    ):
        self._n_features = n_features
        self._spatial_dims = spatial_dims
        self._rho = rho
        self._delta = delta
        self._mu_ma = np.zeros(n_features)
        self._sigma_sq_ma = np.zeros(n_features)
        self._learnable_affine = learnable_affine
    @override
    def get_params(self):
        if not self._learnable_affine:
            return {}
        return {
            "gamma": ParameterSpecification((self._n_features,)),
            "beta": ParameterSpecification((self._n_features,))
        }
    @override
    def _construct(self, input_node: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        input_shape = input_node.get_shape()
        feature_dim_index = len(input_shape) - self._spatial_dims - 1
        if mode == EvaluationMode.INFERENCE:
            mu = self._mu_ma
            sigma_sq = self._sigma_sq_ma
        elif mode == EvaluationMode.TRAINING:
            input_val = input_node.get_value()
            
            mu = input_val.mean(tuple(i for i in range(len(input_shape)) if i != feature_dim_index))
            mu_newaxes = (np.newaxis,) * feature_dim_index + (slice(0, None),) + (np.newaxis,) * self._spatial_dims
            sigma_sq = ((input_val - mu[mu_newaxes]) ** 2).mean(tuple(i for i in range(len(input_shape)) if i != feature_dim_index))
            
            self._mu_ma = (1 - self._rho) * self._mu_ma + self._rho * mu
            self._sigma_sq_ma = (1 - self._rho) * self._sigma_sq_ma + self._rho * sigma_sq

        std_offset_node = nodes.TransposeNode(
            nodes.ExtendNode(
                nodes.ConstantNode(-mu),
                input_shape[:feature_dim_index] + input_shape[(feature_dim_index + 1):]
            ),
            permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
        )
        std_divide_node = nodes.TransposeNode(
            nodes.ExtendNode(
                nodes.ConstantNode((sigma_sq + self._delta) ** 0.5),
                input_shape[:feature_dim_index] + input_shape[(feature_dim_index + 1):]
            ),
            permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
        )
        standardized_node = nodes.ElementwiseNode(
            elementwise.ElementwiseMul(2), [
                nodes.ElementwiseNode(
                    elementwise.ElementwiseAdd(2), [
                        input_node,
                        std_offset_node
                    ]
                ),
                nodes.ElementwiseNode(
                    elementwise.ElementwisePow(-1), [
                        std_divide_node
                    ]
                )
            ]
        )
        
        if not self._learnable_affine:
            return ComputationalGraph(
                output_node=standardized_node,
                param_nodes={}
            )
        
        gamma_node = nodes.ConstantNode(params["gamma"])
        beta_node = nodes.ConstantNode(params["beta"])
        multiplier_node = nodes.TransposeNode(
            nodes.ExtendNode(
                gamma_node,
                input_shape[:feature_dim_index] + input_shape[(feature_dim_index + 1):]
            ),
            permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
        )
        shift_node = nodes.TransposeNode(
            nodes.ExtendNode(
                beta_node,
                input_shape[:feature_dim_index] + input_shape[(feature_dim_index + 1):]
            ),
            permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
        )
        return ComputationalGraph(
            output_node=nodes.ElementwiseNode(
                elementwise.ElementwiseAdd(2), [
                    nodes.ElementwiseNode(
                        elementwise.ElementwiseMul(2), [
                            multiplier_node,
                            standardized_node,
                        ]
                    ),
                    shift_node
                ]
            ),
            param_nodes={
                "gamma": gamma_node,
                "beta": beta_node,
            }
        )
