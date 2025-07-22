
from typing import *
import numpy as np
import nodes
import elementwise
from neural_network import ComputationalGraph
from loss.loss_function import LossFunction

class MSELoss(LossFunction):
    """
    Computes the MSE between the model output and target values.
    The usual choice for training networks for regression tasks.

    For the final loss to be a scalar (required for training), the model output must have only one feature.
    """

    @override
    def _construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        assert graph.output_node.get_shape() == target.shape
        target_node = nodes.ConstantNode(target)
        sq_diff_node = nodes.ElementwiseNode(elementwise.ElementwiseSquaredDifference(), [graph, target_node])
        mean_node = nodes.AvgNode(sq_diff_node, 1)
        return mean_node
