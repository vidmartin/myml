
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
import nodes
from neural_network import ComputationalGraph, NeuralNetwork, EvaluationMode
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
    def step(self, input: TInput, target: np.ndarray) -> OptimizationStepRelevantInfo:
        """
        Computes the loss and its gradients with respect to parameters and
        then perfoms the training step i.e. updates the parameters accordingly.

        Internally just calls `prepare_step` and then passes the returned object
        to `perform_step`. The same object is then returned from this method.
        """
        relevant_info = self.prepare_step(input, target)
        self.perform_step(relevant_info)
        return relevant_info
    def prepare_step(self, input: TInput, target: np.ndarray) -> OptimizationStepRelevantInfo:
        """
        Computes the loss and its gradients with respect to parameters and
        returns an object, that can be passed to the `perform_step` method
        to perform the training step i.e. update the parameters accordingly.
        """
        graph = self._neural_network.construct(input, self._param_values, EvaluationMode.TRAINING)
        loss_node = self._loss_function.construct(graph, target, EvaluationMode.TRAINING)
        grads_list = loss_node.get_gradients_against(
            [graph.param_nodes[key] for key in self._param_ordering]
        )
        grads_dict = {
            key: val for key, val in zip(self._param_ordering, grads_list)
        }

        relevant_info = OptimizationStepRelevantInfo(input, target, graph, loss_node, grads_dict)
        return relevant_info
    @abstractmethod
    def perform_step(self, relevant_info: OptimizationStepRelevantInfo[TInput]):
        """Updates the parameters according to the given information."""
        raise NotImplementedError()
