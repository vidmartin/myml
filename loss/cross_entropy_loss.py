
from typing import *
import numpy as np
import nodes
from neural_network import ComputationalGraph
from loss.loss_function import LossFunction

class CrossEntropyLoss(LossFunction):
    """
    Computes the cross entropy between the output logits and target probabilities
    and performs the mean along the first dimension.
    The usual choice for training networks for classification tasks.

    If p_i are the target probabilities and q_i the probabilities predicted by the model,
    the cross entropy is defined as $-\sum_i p_i \log q_i$.

    If your dataset contains just target class labels instead of probabilities,
    consider wrapping this instance in `OneHotEncodeWrapLoss`, or transforming
    the targets into probabilities yourself e.g. using `utils.one_hot_encode`.
    """

    @override
    def _construct(self, graph: ComputationalGraph, target: np.ndarray) -> nodes.TensorNode:
        assert graph.output_node.get_shape() == target.shape
        target_node = nodes.ConstantNode(target)
        cross_entropy_node = nodes.CrossEntropyLogitsNode(graph.output_node, target_node)
        mean_node = nodes.AvgNode(cross_entropy_node, 1)
        return mean_node
