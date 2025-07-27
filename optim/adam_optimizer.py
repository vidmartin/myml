
from typing import *
import numpy as np
from optim.neural_network_optimizer import NeuralNetworkOptimizer
from optim.defaults import LR_DEFAULT, RHO1_DEFAULT, RHO2_DEFAULT, DELTA_DEFAULT
from loss.loss_function import LossFunction
from neural_network import NeuralNetwork

TInput = TypeVar("TInput")

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
        weight_decay: float = 0.0,
    ):
        super().__init__(neural_network, loss_function, init_params)
        self._lr = lr
        self._rho1 = rho1
        self._rho2 = rho2
        self._delta = delta
        self._weight_decay = weight_decay

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
    def perform_step(self, relevant_info):
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
            self._param_values[key] -= self._lr * self._weight_decay * self._param_values[key]
