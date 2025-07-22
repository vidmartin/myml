
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
import nodes
from neural_network import ComputationalGraph, NeuralNetwork
from loss.loss_function import LossFunction

TInput = TypeVar("TInpput")

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
