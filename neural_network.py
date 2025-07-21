
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
import nodes

@dataclasses.dataclass
class ParameterSpecification:
    shape: tuple[int, ...]

@dataclasses.dataclass
class ComputationalGraph:
    output_node: nodes.TensorNode
    param_nodes: Dict[str, nodes.ConstantNode]

TInput = TypeVar("TInput")

class NeuralNetwork(ABC, Generic[TInput]):
    @abstractmethod
    def get_params(self) -> Dict[str, ParameterSpecification]:
        raise NotImplementedError()
    def assert_params_valid(self, params: Dict[str, np.ndarray]):
        param_specs = self.get_params()
        assert param_specs.keys() == params.keys()
        assert all(params[k].shape == param_specs[k].shape for k in params)
    @final
    def construct(self, input: TInput, params: Dict[str, np.ndarray]) -> ComputationalGraph:
        self.assert_params_valid(params)
        return self._construct(input, params)
    @abstractmethod
    def _construct(self, input: TInput, params: Dict[str, np.ndarray]) -> ComputationalGraph:
        raise NotImplementedError()
    
class LossFunction(ABC):
    @final
    def construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        return self._construct(graph, target)
    @abstractmethod
    def _construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        raise NotImplementedError()
    
@dataclasses.dataclass
class OptimizationStepRelevantInfo(Generic[TInput]):
    input: TInput
    target: np.ndarray
    graph: ComputationalGraph
    loss_node: nodes.TensorNode
    grads_dict: Dict[str, np.ndarray]

class NeuralNetworkOptimizer(ABC, Generic[TInput]):
    def __init__(
        self,
        neural_network: NeuralNetwork[TInput],
        loss_function: LossFunction,
        init_params: Dict[str, np.ndarray]
    ):
        neural_network.assert_params_valid(init_params)
        self._neural_network = neural_network
        self._loss_function = loss_function
        self._param_values = init_params
        self._param_ordering = list(self._neural_network.get_params().keys())
    def step(self, input: TInput, target: np.ndarray):
        graph = self._neural_network.construct(input, self._param_values)
        loss_node = self._loss_function.construct(graph, target)
        grads_list = loss_node.get_gradients_against(
            [graph.param_nodes[key] for key in self._param_ordering]
        )
        grads_dict = {
            key: val for key, val in zip(self._param_ordering, grads_list)
        }
        self._step(
            OptimizationStepRelevantInfo(input, target, graph, loss_node, grads_dict)
        )
    @abstractmethod
    def _step(self, relevant_info: OptimizationStepRelevantInfo[TInput]):
        raise NotImplementedError()

LR_DEFAULT = 0.001
MU_DEFAULT = 0.0
RHO_DEFAULT = 0.99
RHO1_DEFAULT = 0.9
RHO2_DEFAULT = 0.999
DELTA_DEFAULT = 1e-8

class SGDOptimizer(NeuralNetworkOptimizer):
    def __init__(
        self,
        neural_network: NeuralNetwork[TInput],
        loss_function: LossFunction,
        init_params: Dict[str, np.ndarray],
        lr: float = LR_DEFAULT,
        mu: float = 0.0,
    ):
        super().__init__(neural_network, loss_function, init_params)
        self._lr = lr
        self._mu = mu
        
        param_specs = neural_network.get_params()
        self._velocities = {
            key: np.zeros(param_specs[key].shape)
            for key in self._param_ordering
        }
    @override
    def _step(self, relevant_info):
        for key in self._param_ordering:
            self._velocities[key] = \
                self._mu * self._velocities[key] + \
                self._lr * relevant_info.grads_dict[key]
            self._param_values[key] += self._velocities[key]
    # TODO: Nesterov

class RMSPropMomentumOptimizer(NeuralNetworkOptimizer):
    def __init__(
        self,
        neural_network: NeuralNetwork[TInput],
        loss_function: LossFunction,
        init_params: Dict[str, np.ndarray],
        lr: float = LR_DEFAULT,
        mu: float = MU_DEFAULT,
        rho: float = RHO_DEFAULT,
        delta: float = DELTA_DEFAULT,
    ):
        super().__init__(neural_network, loss_function, init_params)
        self._lr = lr
        self._mu = mu
        self._rho = rho
        self._delta = delta

        param_specs = neural_network.get_params()
        self._velocities = {
            key: np.zeros(param_specs[key].shape)
            for key in self._param_ordering
        }
        self._aggregates = {
            key: np.zeros(param_specs[key].shape)
            for key in self._param_ordering
        }
    @override
    def _step(self, relevant_info):
        for key in self._param_ordering:
            self._aggregates[key] = \
                self._rho * self._aggregates[key] + \
                (1 - self._rho) * relevant_info.grads_dict[key] ** 2
            grad_multiplier = self._lr / (self._aggregates[key] ** 0.5 + self._delta)
            self._velocities[key] = \
                self._mu * self._velocities[key] + \
                grad_multiplier * relevant_info.grads_dict[key]
            self._param_values[key] += self._velocities[key]

class AdamOptimizer(NeuralNetworkOptimizer):
    def __init__(
        self,
        neural_network: NeuralNetwork[TInput],
        loss_function: LossFunction,
        init_params: Dict[str, np.ndarray],
        lr: float = LR_DEFAULT,
        rho1: float = RHO1_DEFAULT,
        rho2: float = RHO2_DEFAULT,
        delta: float = DELTA_DEFAULT,
    ):
        super().__init__(neural_network, loss_function, init_params)
        self._lr = lr
        self._rho1 = rho1
        self._rho2 = rho2
        self._delta = delta

        param_specs = neural_network.get_params()
        self._aggregates1 = {
            key: np.zeros(param_specs[key].shape)
            for key in self._param_ordering
        }
        self._aggregates2 = {
            key: np.zeros(param_specs[key].shape)
            for key in self._param_ordering
        }
        self._t: int = 0
    @override
    def _step(self, relevant_info):
        self._t += 1
        for key in self._param_ordering:
            self._aggregates1[key] = \
                self._rho1 * self._aggregates1[key] + \
                (1 - self._rho1) * relevant_info.grads_dict[key]
            self._aggregates2[key] = \
                self._rho2 * self._aggregates2[key] + \
                (1 - self._rho2) * relevant_info.grads_dict[key] ** 2

            corrected_aggregate1 = self._aggregates1[key] / (1 - self._rho1 ** self._t)
            corrected_aggregate2 = self._aggregates2[key] / (1 - self._rho2 ** self._t)
            
            self._param_values[key] -= self._lr * corrected_aggregate1 / (self._delta + corrected_aggregate2 ** 0.5)
