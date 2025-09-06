
from typing import *
from node_visitor.node_visitor import NodeVisitor
import nodes

T = TypeVar("T", bound=nodes.TensorNode)

class TypeFinderNodeVisitor(NodeVisitor[set[T]], Generic[T]):
    def __init__(self, target_type: Type[T]):
        self._target_type = target_type
    def _inspect_deps(self, node: nodes.LazyDependentNode) -> set[T]:
        return set().union(*[
            dep.accept(self) for dep in node._deps
        ])
    @override
    def visit_constant_node(self, node: nodes.ConstantNode) -> set[T]:
        return { node } if self._target_type is nodes.ConstantNode else set()
    @override
    def visit_elementwise_node(self, node: nodes.ElementwiseNode) -> set[T]:
        temp = { node } if self._target_type is nodes.ElementwiseNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_tensor_dot_node(self, node: nodes.TensorDotNode) -> set[T]:
        temp = { node } if self._target_type is nodes.TensorDotNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_transpose_node(self, node: nodes.TransposeNode) -> set[T]:
        temp = { node } if self._target_type is nodes.TransposeNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_extend_node(self, node: nodes.ExtendNode) -> set[T]:
        temp = { node } if self._target_type is nodes.ExtendNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_sum_node(self, node: nodes.SumNode) -> set[T]:
        temp = { node } if self._target_type is nodes.SumNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_logsumexp_node(self, node: nodes.LogSumExpNode) -> set[T]:
        temp = { node } if self._target_type is nodes.LogSumExpNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_softmax_node(self, node: nodes.SoftmaxNode) -> set[T]:
        temp = { node } if self._target_type is nodes.SoftmaxNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_avg_node(self, node: nodes.AvgNode) -> set[T]:
        temp = { node } if self._target_type is nodes.AvgNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_cross_entropy_logits_node(self, node: nodes.CrossEntropyLogitsNode) -> set[T]:
        temp = { node } if self._target_type is nodes.CrossEntropyLogitsNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_reshape_node(self, node: nodes.ReshapeNode) -> set[T]:
        temp = { node } if self._target_type is nodes.ReshapeNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_convolution_node(self, node: nodes.ConvolutionNode) -> set[T]:
        temp = { node } if self._target_type is nodes.ConvolutionNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_max_pool_node(self, node: nodes.MaxPoolNode) -> set[T]:
        temp = { node } if self._target_type is nodes.MaxPoolNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_slice_node(self, node: nodes.SliceNode) -> set[T]:
        temp = { node } if self._target_type is nodes.SliceNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_stack_node(self, node: nodes.StackNode) -> set[T]:
        temp = { node } if self._target_type is nodes.StackNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_multichannel_convolution_node(self, node: nodes.MultichannelConvolutionNode) -> set[T]:
        temp = { node } if self._target_type is nodes.MultichannelConvolutionNode else set()
        return temp | self._inspect_deps(node)
    @override
    def visit_batch_norm_node(self, node: nodes.BatchNormNode) -> set[T]:
        temp = { node } if self._target_type is nodes.BatchNormNode else set()
        return temp | self._inspect_deps(node)
