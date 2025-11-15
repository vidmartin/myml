
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
        self._sigma_sq_ma = np.ones(n_features)
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
        pass
    def _update_moving_averages(self, input_shape: tuple[int, ...], mu: np.ndarray, proto_sigma_sq: np.ndarray):
        feature_dim_index, other_axes = self._get_feature_dim_index_and_other_axes(input_shape)
        self._mu_ma = (1 - self._rho) * self._mu_ma + self._rho * mu
        sigma_sq_divisor = functools.reduce(lambda x, y: x * y, [input_shape[i] for i in other_axes], 1) - 1
        sigma_sq_unbiased = proto_sigma_sq.sum(other_axes) / sigma_sq_divisor
        self._sigma_sq_ma = (1 - self._rho) * self._sigma_sq_ma + self._rho * sigma_sq_unbiased
    @override
    def _construct(self, input_node: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode, metadata: dict[str, Any]) -> ComputationalGraph:
        input_shape = input_node.get_shape()
        feature_dim_index, other_axes = self._get_feature_dim_index_and_other_axes(input_shape)
        if mode == EvaluationMode.INFERENCE:
            mu_node = nodes.ConstantNode(self._mu_ma)
            sigma_sq_node = nodes.ConstantNode(self._sigma_sq_ma)
        elif mode == EvaluationMode.TRAINING:
            input_val = input_node.get_value()

            mu_node = nodes.AvgNode(
                nodes.TransposeNode(input_node, permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape))),
                len(input_shape) - 1
            )
            mu_node_extended = nodes.TransposeNode(
                nodes.ExtendNode(mu_node, tuple(input_shape[a] for a in other_axes)),
                permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
            )
            proto_sigma_sq_node = nodes.ElementwiseNode(
                elementwise.ElementwiseIPow(2), [
                    nodes.ElementwiseNode(
                        elementwise.ElementwiseAdd(2), [
                            input_node,
                            nodes.ElementwiseNode(
                                elementwise.ElementwiseScale(-1.0), [
                                    mu_node_extended
                                ]
                            )
                        ]
                    )
                ]
            )
            sigma_sq_node = nodes.AvgNode(
                nodes.TransposeNode(proto_sigma_sq_node, permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape))),
                len(input_shape) - 1,
            ) # computing arithmetic average i.e. we have biased estimate of variance

            self._update_moving_averages(input_shape, mu_node.get_value(), proto_sigma_sq_node.get_value())

        std_offset_node = nodes.TransposeNode(
            nodes.ExtendNode(
                nodes.ElementwiseNode(
                    elementwise.ElementwiseScale(-1.0),
                    [mu_node],
                ),
                input_shape[:feature_dim_index] + input_shape[(feature_dim_index + 1):]
            ),
            permutation.Permutation.bring_to_back((feature_dim_index,), len(input_shape)).inverse()
        )
        std_divide_node = nodes.TransposeNode(
            nodes.ExtendNode(
                nodes.ElementwiseNode(
                    elementwise.ElementwiseFPow(0.5), [
                        nodes.ElementwiseNode(
                            elementwise.ElementwiseAdd(2), [
                                sigma_sq_node,
                                nodes.ExtendNode(
                                    nodes.ConstantNode(np.array(self._delta)),
                                    sigma_sq_node.get_shape()
                                )
                            ]
                        )
                    ]
                ),
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
                    elementwise.ElementwiseIPow(-1), [
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
                            standardized_node,
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
