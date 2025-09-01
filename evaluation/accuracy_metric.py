
from typing import *
import dataclasses
from evaluation.metric import Metric

@dataclasses.dataclass
class AccuracyMetricVars:
    n_samples_correct: int
    n_samples_total: int

class AccuracyMetric(Metric[AccuracyMetricVars]):
    @override
    def get_name(self):
        return "accuracy"
    @override
    def init_vars(self):
        return AccuracyMetricVars(0, 0)
    @override
    def process_batch(self, vars, features, outputs, targets):
        assert len(targets.shape) == 1
        assert len(outputs.shape) == 2
        n_correct = (outputs.argmax(1) == targets).sum()
        return AccuracyMetricVars(
            n_samples_correct=vars.n_samples_correct + n_correct,
            n_samples_total=vars.n_samples_total + features.shape[0],
        )
    @override
    def get_result(self, vars):
        return vars.n_samples_correct / vars.n_samples_total
