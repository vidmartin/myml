
from typing import *
import numpy as np
import nodes
from neural_network import ComputationalGraph, EvaluationMode
from loss.loss_function import LossFunction

class LabelSmoothingWrapLoss(LossFunction):
    """
    Wrapps a `LossFunction` instance and performs the label smoothing regularization technique
    (assuming that the targets passed are probabilities of class labels).
    """

    def __init__(self, wrapped: LossFunction, alpha: float):
        self._wrapped = wrapped
        self._alpha = alpha
    @override
    def _construct(self, graph: ComputationalGraph, target: np.ndarray, mode: EvaluationMode) -> nodes.TensorNode:
        smoothed_target = self._alpha / target.shape[-1] + (1 - self._alpha) * target
        return self._wrapped.construct(graph, smoothed_target)
