
from typing import *
import numpy as np
from optim.neural_network_optimizer import NeuralNetworkOptimizer
from optim.defaults import LR_DEFAULT, MU_DEFAULT, RHO_DEFAULT, DELTA_DEFAULT
from loss.loss_function import LossFunction
from neural_network import NeuralNetwork

TInput = TypeVar("TInput")

class RMSPropOptimizer(NeuralNetworkOptimizer):
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
    def perform_step(self, relevant_info):
        for key in self._param_ordering:
            self._aggregates[key] = \
                self._rho * self._aggregates[key] + \
                (1 - self._rho) * relevant_info.grads_dict[key] ** 2
            grad_multiplier = self._lr / (self._aggregates[key] ** 0.5 + self._delta)
            self._velocities[key] = \
                self._mu * self._velocities[key] - \
                grad_multiplier * relevant_info.grads_dict[key]
            self._param_values[key] += self._velocities[key]
