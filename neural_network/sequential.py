
from typing import *
import numpy as np
from neural_network.neural_network import NeuralNetwork, ComputationalGraph, EvaluationMode
import nodes

class SequentialModule(NeuralNetwork[nodes.TensorNode]):
    def __init__(self, children: list[NeuralNetwork[nodes.TensorNode]]):
        self._children = children
    @override
    def get_params(self):
        return {
            f"{i}.{name}": param
            for i, child in enumerate(self._children)
            for name, param in child.get_params().items()
        }
    @override
    def _construct(self, input: nodes.TensorNode, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        output_node = input
        param_nodes: dict[str, nodes.ConstantNode] = {}
        for i, child in enumerate(self._children):
            param_prefix = f"{i}."
            relevant_params = {
                name[len(param_prefix):]: param
                for name, param in params.items()
                if name.startswith(param_prefix)
            }
            child_graph = child.construct(output_node, relevant_params, mode)
            param_nodes = param_nodes | {
                f"{param_prefix}{name}": param_node
                for name, param_node in child_graph.param_nodes.items()
            }
            output_node = child_graph.output_node
        return ComputationalGraph(
            output_node=output_node,
            param_nodes=param_nodes,
        )
