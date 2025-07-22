
from typing import *
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import nodes

TReturn = TypeVar("TReturn")

class NodeVisitor(ABC, Generic[TReturn]):
    @abstractmethod
    def visit_constant_node(self, node: nodes.ConstantNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_elementwise_node(self, node: nodes.ElementwiseNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_tensor_dot_node(self, node: nodes.TensorDotNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_transpose_node(self, node: nodes.TransposeNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_extend_node(self, node: nodes.ExtendNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_sum_node(self, node: nodes.SumNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_logsumexp_node(self, node: nodes.LogSumExpNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_softmax_node(self, node: nodes.SoftmaxNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_avg_node(self, node: nodes.AvgNode) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_cross_entropy_logits_node(self, node: nodes.CrossEntropyLogitsNode) -> TReturn:
        raise NotImplementedError()
