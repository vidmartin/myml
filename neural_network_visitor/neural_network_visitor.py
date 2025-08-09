
from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    import neural_network

TResult = TypeVar("TResult")

class NeuralNetworkVisitor(ABC, Generic[TResult]):
    @abstractmethod
    def visit_batch_norm(self, module: neural_network.BatchNormModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_dropout(self, module: neural_network.DropoutModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_elementwise(self, module: neural_network.ElementwiseModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_flatten(self, module: neural_network.FlattenModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_input_numpy(self, module: neural_network.InputNumpyModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_linear(self, module: neural_network.LinearModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_multichannel_convolution_v2(self, module: neural_network.MultichannelConvolutionV2Module) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_multichannel_convolution(self, module: neural_network.MultichannelConvolutionModule) -> TResult:
        raise NotImplementedError()
    @abstractmethod
    def visit_sequential(self, module: neural_network.SequentialModule) -> TResult:
        raise NotImplementedError()
