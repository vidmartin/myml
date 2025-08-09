
from typing import *
import numpy as np
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor
from neural_network.neural_network import ParameterSpecification
import neural_network

class ParamInitializationNeuralNetworkVisitor(NeuralNetworkVisitor[Dict[str, np.ndarray]]):
    def __init__(
        self,
        rng: np.random.Generator,
        linear_weight: Callable[[ParameterSpecification], np.ndarray] | None = None,
        linear_bias: Callable[[ParameterSpecification], np.ndarray] | None = None,
        multichannel_convolution_kernels: Callable[[ParameterSpecification], np.ndarray] | None = None,
        multichannel_convolution_bias: Callable[[ParameterSpecification], np.ndarray] | None = None,
        batch_norm_gamma: Callable[[ParameterSpecification], np.ndarray] | None = None,
        batch_norm_beta: Callable[[ParameterSpecification], np.ndarray] | None = None,
    ):
        self._rng = rng
        self._linear_weight = linear_weight
        self._linear_bias = linear_bias
        self._multichannel_convolution_kernels = multichannel_convolution_kernels
        self._multichannel_convolution_bias = multichannel_convolution_bias
        self._batch_norm_gamma = batch_norm_gamma
        self._batch_norm_beta = batch_norm_beta
    @override
    def visit_batch_norm(self, module: neural_network.BatchNormModule):
        params = module.get_params()
        gamma_spec, beta_spec = [params[s] for s in ("gamma", "beta")]
        return {
            "gamma": np.ones(gamma_spec.shape) \
                if self._batch_norm_gamma is None else self._batch_norm_gamma(gamma_spec),
            "beta": np.zeros(beta_spec.shape) \
                if self._batch_norm_beta is None else self._batch_norm_beta(beta_spec),
        }
    @override
    def visit_dropout(self, module: neural_network.DropoutModule):
        return {}
    @override
    def visit_elementwise(self, module: neural_network.ElementwiseModule):
        return {}
    @override
    def visit_flatten(self, module: neural_network.FlattenModule):
        return {}
    @override
    def visit_input_numpy(self, module: neural_network.InputNumpyModule):
        return module._wrapped.accept(self)
    @override
    def visit_linear(self, module: neural_network.LinearModule):
        params = module.get_params()
        weight_spec, bias_spec = [params[s] for s in ("weight", "bias")]
        return {
            "weight": (self._rng.random(weight_spec.shape) * 2 - 1) * (6.0 / sum(weight_spec.shape)) ** 0.5 \
                if self._linear_weight is None else self._linear_weight(weight_spec),
            "bias": np.zeros(bias_spec.shape) \
                if self._linear_bias is None else self._linear_bias(bias_spec)
        }
    @override
    def visit_multichannel_convolution(self, module: neural_network.MultichannelConvolutionModule):
        params = module.get_params()
        kernels_spec, bias_spec = [params[s] for s in ("kernels", "bias")]
        return {
            "kernels": (self._rng.random(kernels_spec.shape) * 2 - 1) * (6.0 / sum(kernels_spec.shape)) ** 0.5 \
                if self._multichannel_convolution_kernels is None else self._multichannel_convolution_kernels(kernels_spec),
            "bias": np.zeros(bias_spec.shape) \
                if self._multichannel_convolution_bias is None else self._multichannel_convolution_bias(bias_spec)
        }
    @override
    def visit_multichannel_convolution_v2(self, module: neural_network.MultichannelConvolutionV2Module):
        params = module.get_params()
        kernels_spec, bias_spec = [params[s] for s in ("kernels", "bias")]
        return {
            "kernels": (self._rng.random(kernels_spec.shape) * 2 - 1) * (6.0 / sum(kernels_spec.shape)) ** 0.5 \
                if self._multichannel_convolution_kernels is None else self._multichannel_convolution_kernels(kernels_spec),
            "bias": np.zeros(bias_spec.shape) \
                if self._multichannel_convolution_bias is None else self._multichannel_convolution_bias(bias_spec)
        }
    @override
    def visit_sequential(self, module: neural_network.SequentialModule):
        return {
            f"{i}.{name}": value
            for i, child in enumerate(module._children)
            for name, value in child.accept(self).items()
        }
