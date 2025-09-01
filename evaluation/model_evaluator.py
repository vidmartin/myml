
from typing import *
import dataclasses
import numpy as np
import neural_network
import data_utils
from evaluation.metric import Metric

@dataclasses.dataclass
class ModelEvaluationResult:
    metric_values: dict[Metric, Any]

class ModelEvaluator:
    def __init__(self, metrics: list[Metric]):
        self._metrics = metrics
    def evaluate(self, model: neural_network.NeuralNetwork, model_params: dict[str, np.ndarray], data: data_utils.Dataloader) -> ModelEvaluationResult:
        metric_vars = [met.init_vars() for met in self._metrics]
        for X, Y in data:
            graph = model.construct(X, model_params)
            output = graph.output_node.get_value()
            metric_vars = [
                met.process_batch(vars, X, output, Y)
                for met, vars in zip(self._metrics, metric_vars)
            ]
        return ModelEvaluationResult(
            metric_values={
                met: met.get_result(vars)
                for met, vars in zip(self._metrics, metric_vars)
            }
        )
