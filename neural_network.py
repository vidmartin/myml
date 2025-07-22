
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
import nodes

@dataclasses.dataclass
class ParameterSpecification:
    shape: tuple[int, ...]

@dataclasses.dataclass
class ComputationalGraph:
    output_node: nodes.TensorNode
    param_nodes: Dict[str, nodes.ConstantNode]

TInput = TypeVar("TInput")

class NeuralNetwork(ABC, Generic[TInput]):
    @abstractmethod
    def get_params(self) -> Dict[str, ParameterSpecification]:
        raise NotImplementedError()
    def assert_params_valid(self, params: Dict[str, np.ndarray]):
        param_specs = self.get_params()
        assert param_specs.keys() == params.keys()
        assert all(params[k].shape == param_specs[k].shape for k in params)
    @final
    def construct(self, input: TInput, params: Dict[str, np.ndarray]) -> ComputationalGraph:
        self.assert_params_valid(params)
        return self._construct(input, params)
    @abstractmethod
    def _construct(self, input: TInput, params: Dict[str, np.ndarray]) -> ComputationalGraph:
        raise NotImplementedError()
