
from typing import *
import dataclasses
import nodes
from neural_network import ComputationalGraph
from loss import LossFunction
from evaluation.metric import Metric

@dataclasses.dataclass
class AvgLossMetricVars:
    total_loss: float
    n_batches: int

class AvgLossMetric(Metric[AvgLossMetricVars]):
    def __init__(self, loss_function: LossFunction):
        self._loss_function = loss_function
    @override
    def get_name(self):
        return "avg_loss"
    @override
    def init_vars(self):
        return AvgLossMetricVars(0.0, 0)
    @override
    def process_batch(self, vars, features, outputs, targets):
        graph = ComputationalGraph(
            output_node=nodes.ConstantNode(outputs),
            param_nodes={}
        )
        loss_node = self._loss_function.construct(graph, targets)
        loss_val = loss_node.get_value().item()
        return AvgLossMetricVars(
            vars.total_loss + loss_val,
            vars.n_batches + 1,
        )
    @override
    def get_result(self, vars):
        return vars.total_loss / vars.n_batches
