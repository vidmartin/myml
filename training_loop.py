
from typing import *
import dataclasses
import enum
from neural_network import NeuralNetwork
from optim import NeuralNetworkOptimizer
from data_utils import Dataloader
from evaluation import ModelEvaluationResult, ModelEvaluator

@dataclasses.dataclass
class BatchInfo:
    epoch_index: int
    batch_index: int
    total_batches: int
    loss: float

@dataclasses.dataclass
class EpochInfo:
    epoch_index: int
    eval_result: ModelEvaluationResult

class EpochCallbackResponse(enum.Enum):
    CONTINUE = 0
    BREAK = 1

def default_batch_callback(batch_info: BatchInfo):
    print(f"\r   batch {batch_info.batch_index + 1}/{batch_info.total_batches}, loss: {batch_info.loss:.3f}", end="")

def default_epoch_callback(epoch_info: EpochInfo):
    metrics_string = ", ".join(
        f"{dataset_name}_{metric.get_name()}" + "=" + metric.format_result(val)
        for dataset_name, metric_dict in epoch_info.eval_result.metric_values.items()
        for metric, val in metric_dict.items()
    )
    print(f"\rEpoch {epoch_info.epoch_index + 1}: {metrics_string}")
    return EpochCallbackResponse.CONTINUE

TRAIN_DATALOADER_KEY: str = "train"

class TrainingLoop:
    def __init__(
        self,
        optimizer: NeuralNetworkOptimizer,
        dataloaders: dict[str, Dataloader],
        evaluator: ModelEvaluator,
        max_epochs: int | None,
        batch_callback: Callable[[BatchInfo], None] = default_batch_callback,
        epoch_callback: Callable[[EpochInfo], None | EpochCallbackResponse] = default_epoch_callback,
    ):
        self._optimizer = optimizer
        assert TRAIN_DATALOADER_KEY in dataloaders, \
            f"the '{TRAIN_DATALOADER_KEY}' key must be present in the `dataloaders` dict!"
        self._dataloaders = dataloaders
        self._evaluator = evaluator
        self._max_epochs = max_epochs
        self._batch_callback = batch_callback
        self._epoch_callback = epoch_callback
    def run(self):
        train_dataloader = self._dataloaders[TRAIN_DATALOADER_KEY]
        for i in range(self._max_epochs):
            for j, (X, Y) in enumerate(train_dataloader):
                step_res = self._optimizer.step(X, Y)
                loss_val = step_res.loss_node.get_value().item()
                self._batch_callback(
                    BatchInfo(i, j, len(train_dataloader), loss_val)
                )
            eval_result = self._evaluator.evaluate(
                self._optimizer.get_model(),
                self._optimizer.get_param_values(),
                self._dataloaders
            )
            epoch_callback_response = self._epoch_callback(
                EpochInfo(i, eval_result)
            )
            if epoch_callback_response == EpochCallbackResponse.BREAK:
                break
