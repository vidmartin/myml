
from typing import *
import numpy as np
import nodes
import elementwise
from neural_network import ComputationalGraph
from loss.loss_function import LossFunction

class WeightPenalizeWrapLoss(LossFunction):
    """
    A wrapper around a `LossFunction` instance that adds penalization of the norms of parameters.

    You can specify which parameters should be penalized with the constructor arguments.
    Note that it is unwise to penalize biases.
    """
    def __init__(
        self,
        wrapped: LossFunction,
        parameter_selector: Callable[[str], bool],
        factor: float,
        kind: Literal["l1"] | Literal["l2"]
    ):
        self._wrapped = wrapped
        self._parameter_selector = parameter_selector
        self._factor = factor
        self._kind = kind

    @override
    def _construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        param_nodes = [
            param_node for param_name, param_node in graph.param_nodes.items()
            if self._parameter_selector(param_name)
        ]

        penalizing_nodes: list[nodes.TensorNode]
        if self._kind == "l1":
            penalizing_nodes = [
                nodes.ElementwiseNode(
                    elementwise.ElementwiseAbs(), [node]
                ) for node in param_nodes
            ]
        elif self._kind == "l2":
            penalizing_nodes = [
                nodes.ElementwiseNode(
                    elementwise.ElementwisePow(2), [node]
                ) for node in param_nodes
            ]
        else:
            raise NotImplementedError(f"unknown weight penalization kind '{self._kind}'")
        
        penalizing_nodes = [
            nodes.SumNode(node, len(node.get_shape()))
            for node in penalizing_nodes
        ]

        total_penalizing_node = nodes.ElementwiseNode(
            elementwise.ElementwiseScale(self._factor), [
                nodes.ElementwiseNode(
                    elementwise.ElementwiseAdd(len(penalizing_nodes)),
                    penalizing_nodes
                )
            ]
        )

        return nodes.ElementwiseNode(
            elementwise.ElementwiseAdd(2), [
                total_penalizing_node,
                self._wrapped.construct(graph, target)
            ]
        )
