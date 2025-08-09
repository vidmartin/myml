
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import enum
import numpy as np
import nodes
from neural_network_visitor.neural_network_visitor import NeuralNetworkVisitor

@dataclasses.dataclass
class ParameterSpecification:
    shape: tuple[int, ...]

@dataclasses.dataclass
class ComputationalGraph:
    output_node: nodes.TensorNode
    param_nodes: Dict[str, nodes.ConstantNode]

class EvaluationMode(enum.Enum):
    TRAINING = 0
    INFERENCE = 1

TInput = TypeVar("TInput")
TResult = TypeVar("TResult")

class NeuralNetwork(ABC, Generic[TInput]):
    @abstractmethod
    def get_params(self) -> Dict[str, ParameterSpecification]:
        raise NotImplementedError()
    def assert_params_valid(self, params: Dict[str, np.ndarray]):
        param_specs = self.get_params()
        assert param_specs.keys() == params.keys()
        assert all(params[k].shape == param_specs[k].shape for k in params), \
            f"the following params were passed wrong shapes: " + \
            "; ".join(
                f"param {k} wants shape {param_specs[k].shape}, but got {params[k].shape}"
                for k in params if param_specs[k].shape != params[k].shape
            )
    @final
    def construct(self, input: TInput, params: Dict[str, np.ndarray], mode: EvaluationMode = EvaluationMode.INFERENCE) -> ComputationalGraph:
        self.assert_params_valid(params)
        return self._construct(input, params, mode)
    @abstractmethod
    def _construct(self, input: TInput, params: Dict[str, np.ndarray], mode: EvaluationMode) -> ComputationalGraph:
        raise NotImplementedError()
    @abstractmethod
    def accept(self, visitor: NeuralNetworkVisitor[TResult]) -> TResult:
        raise NotImplementedError()
