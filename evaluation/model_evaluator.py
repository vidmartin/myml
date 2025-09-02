
from typing import *
import dataclasses
import numpy as np
import neural_network
import data_utils
from evaluation.metric import Metric

@dataclasses.dataclass
class ModelEvaluationResult:
    metric_values: dict[str, dict[Metric, float]]

class ModelEvaluator:
    def __init__(self, metrics: list[Metric]):
        self._metrics = metrics
    def evaluate(
        self,
        model: neural_network.NeuralNetwork,
        model_params: dict[str, np.ndarray],
        dataloaders: dict[str, data_utils.Dataloader]
    ) -> ModelEvaluationResult:
        metric_vars = {
            key: [met.init_vars() for met in self._metrics]
            for key in dataloaders.keys()
        }
        for key, dataloader in dataloaders.items():
            for X, Y in dataloader:
                graph = model.construct(X, model_params)
                output = graph.output_node.get_value()
                metric_vars[key] = [
                    met.process_batch(vars, X, output, Y)
                    for met, vars in zip(self._metrics, metric_vars[key])
                ]
        return ModelEvaluationResult(
            metric_values={
                key: {
                    met: met.get_result(vars)
                    for met, vars in zip(self._metrics, vars_list)
                }
                for key, vars_list in metric_vars.items()
            }
        )
