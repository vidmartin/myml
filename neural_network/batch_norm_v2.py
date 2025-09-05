
from typing import *
import functools
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, ParameterSpecification, EvaluationMode
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
import nodes
import elementwise
import permutation

TResult = TypeVar("TResult")

GAMMA_PARAM_NAME = "weight"
BETA_PARAM_NAME = "bias"

class BatchNormV2Module(NeuralNetwork[nodes.TensorNode]):
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
            GAMMA_PARAM_NAME: ParameterSpecification((self._n_features,)),
            BETA_PARAM_NAME: ParameterSpecification((self._n_features,))
        }
    def _get_feature_dim_index_and_other_axes(self, input_shape: tuple[int, ...]):
        """Returns a tuple `(feature_dim_index, other_axes)`, where `other_axes` is a tuple like `(0, 1, ..., feature_dim_index - 1, feature_dim_index + 1, ..., len(input_shape))`."""
        feature_dim_index = len(input_shape) - self._spatial_dims - 1
        other_axes = tuple(i for i in range(len(input_shape)) if i != feature_dim_index)
        return feature_dim_index, other_axes
    # def _update_moving_averages(self, input_shape: tuple[int, ...], mu: np.ndarray, proto_sigma_sq: np.ndarray):
    #     feature_dim_index, other_axes = self._get_feature_dim_index_and_other_axes(input_shape)
    #     self._mu_ma = (1 - self._rho) * self._mu_ma + self._rho * mu
    #     sigma_sq_divisor = functools.reduce(lambda x, y: x * y, [input_shape[i] for i in other_axes], 1) - 1
    #     sigma_sq_unbiased = proto_sigma_sq.sum(other_axes) / sigma_sq_divisor
    #     self._sigma_sq_ma = (1 - self._rho) * self._sigma_sq_ma + self._rho * sigma_sq_unbiased
    def _broadcast_features(self, arr: np.ndarray, n_dims: int, feature_dim_index: int) -> np.ndarray:
        return arr[(np.newaxis,) * feature_dim_index + (...,)][(...,) + (np.newaxis,) * (n_dims - feature_dim_index - 1)]
    def _update_moving_averages(self, input_node: nodes.TensorNode):
        input_shape = input_node.get_shape()
        input_val = input_node.get_value()
        feature_dim_index, other_axes = self._get_feature_dim_index_and_other_axes(input_shape)
        mu = input_val.mean(other_axes)
        # mu_ext = self._broadcast_features(mu, len(input_shape), feature_dim_index)
        # sigsq = ((input_val - mu_ext)**2).sum(other_axes) / (functools.reduce(lambda x, y: x * y, [input_shape[a] for a in other_axes]) - 1)
        sigsq = input_val.var(other_axes)

        self._mu_ma = (1 - self._rho) * self._mu_ma + self._rho * mu
        self._sigma_sq_ma = (1 - self._rho) * self._sigma_sq_ma + self._rho * sigsq
    @override
    def _construct(self, input_node: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        input_shape = input_node.get_shape()
        feature_dim_index, other_axes = self._get_feature_dim_index_and_other_axes(input_shape)
        input_val = input_node.get_value()
        if mode == EvaluationMode.INFERENCE:
            mu_ext = self._broadcast_features(self._mu_ma, len(input_shape), feature_dim_index)
            sigma_sq_ext = self._broadcast_features(self._sigma_sq_ma, len(input_shape), feature_dim_index)
            batch_norm_node = nodes.ConstantNode((input_val - mu_ext) / (sigma_sq_ext + self._delta) ** 0.5)
        elif mode == EvaluationMode.TRAINING:
            self._update_moving_averages(input_node)

            batch_norm_node = nodes.TransposeNode(
                nodes.BatchNormNode(
                    nodes.TransposeNode(
                        input_node,
                        permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape))
                    ), 
                    n_axes=len(input_shape) - 1,
                    delta=self._delta
                ),
                permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
            )
        
        if not self._learnable_affine:
            return ComputationalGraph(
                output_node=batch_norm_node,
                param_nodes={}
            )
        
        gamma_node = nodes.ConstantNode(params[GAMMA_PARAM_NAME])
        beta_node = nodes.ConstantNode(params[BETA_PARAM_NAME])
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
                            batch_norm_node,
                        ]
                    ),
                    shift_node
                ]
            ),
            param_nodes={
                GAMMA_PARAM_NAME: gamma_node,
                BETA_PARAM_NAME: beta_node,
            }
        )
    @override
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        return visitor.visit_batch_norm(self)
