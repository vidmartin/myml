
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
    print(f"\rEpoch {epoch_info.epoch_index + 1}: {', '.join(met.get_name() + f'={val:.3f}' for met, val in epoch_info.eval_result.metric_values.items())}")
    return EpochCallbackResponse.CONTINUE

class TrainingLoop:
    def __init__(
        self,
        optimizer: NeuralNetworkOptimizer,
        dataloader: Dataloader,
        evaluator: ModelEvaluator,
        max_epochs: int | None,
        batch_callback: Callable[[BatchInfo], None] = default_batch_callback,
        epoch_callback: Callable[[EpochInfo], None | EpochCallbackResponse] = default_epoch_callback,
    ):
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._evaluator = evaluator
        self._max_epochs = max_epochs
        self._batch_callback = batch_callback
        self._epoch_callback = epoch_callback
    def run(self):
        for i in range(self._max_epochs):
            for j, (X, Y) in enumerate(self._dataloader):
                step_res = self._optimizer.step(X, Y)
                loss_val = step_res.loss_node.get_value().item()
                self._batch_callback(
                    BatchInfo(i, j, len(self._dataloader), loss_val)
                )
            eval_result = self._evaluator.evaluate(
                self._optimizer.get_model(),
                self._optimizer.get_param_values(),
                self._dataloader
            )
            epoch_callback_response = self._epoch_callback(
                EpochInfo(i, eval_result)
            )
            if epoch_callback_response == EpochCallbackResponse.BREAK:
                break
