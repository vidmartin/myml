
from typing import *
import numpy as np
import nodes
import utils
from neural_network import ComputationalGraph, EvaluationMode
from loss.loss_function import LossFunction

class OneHotEncodeWrapLoss(LossFunction):
    """
    A wrapper around a `LossFunction` instance that first performs one-hot encoding of the targets.

    Useful namely for classification tasks where probabilities are predicted, e.g. with `CrossEntropyLoss`.
    """

    def __init__(self, wrapped: LossFunction, n_classes: int):
        self._wrapped = wrapped
        self._n_classes = n_classes

    @override
    def _construct(self, graph: ComputationalGraph, target: np.ndarray, mode: EvaluationMode) -> nodes.TensorNode:
        return self._wrapped.construct(graph, utils.one_hot_encode(target, self._n_classes))
