
from typing import *
import functools
from node_visitor.node_visitor import NodeVisitor
from elementwise_visitor.convert_to_torch_elementwise_visitor import ConvertToTorchElementwiseVisitor
import nodes
import elementwise
import utils
import torch

# TODO: make torch optional
class ConvertToTorchNodeVisitor(NodeVisitor[torch.Tensor]):
    """
    Converts our nodes to PyTorch tensors.
    
    Is here only for purposes of testing our framework against PyTorch. Requires the PyTorch optional dependency.

    The returned tensors are memoized, so that it's possible obtain the leaf tensors in
    which gradient was saved after calling `.backward()` on some tensor returned by
    this visitor.
    """

    @override
    @utils.instance_memo
    def visit_constant_node(self, node):
        return torch.tensor(node.get_value(), requires_grad=True)
    @override
    @utils.instance_memo
    def visit_elementwise_node(self, node):
        deps_nodes = node.get_direct_dependencies()
        deps_torch = [dep.accept(self) for dep in deps_nodes]
        elemwise_visitor = ConvertToTorchElementwiseVisitor(deps_torch)
        return node._function.accept(elemwise_visitor)
    @override
    @utils.instance_memo
    def visit_tensor_dot_node(self, node):
        lhs_node, rhs_node = node.get_direct_dependencies()
        lhs_torch, rhs_torch = lhs_node.accept(self), rhs_node.accept(self)
        return torch.tensordot(lhs_torch, rhs_torch, node._n_axes_to_contract)
    @override
    @utils.instance_memo
    def visit_transpose_node(self, node):
        dep_node, = node.get_direct_dependencies()
        dep_torch = dep_node.accept(self)
        return torch.permute(dep_torch, node._permutation.permutation)
    @override
    @utils.instance_memo
    def visit_extend_node(self, node):
        dep_node, = node.get_direct_dependencies()
        dep_torch = dep_node.accept(self)
        indexer = (torch.newaxis,) * len(node._prepend_dims) + (slice(0, None),) * len(dep_torch.shape)
        dummy = torch.zeros(node._prepend_dims + (1,) * len(dep_torch.shape))
        return dep_torch[indexer] + dummy
    @override
    @utils.instance_memo
    def visit_sum_node(self, node):
        dep_node, = node.get_direct_dependencies()
        dep_torch = dep_node.accept(self)
        return dep_torch.sum(tuple(range(node._n_axes_to_sum)))
    @override
    @utils.instance_memo
    def visit_logsumexp_node(self, node):
        dep_node, = node.get_direct_dependencies()
        dep_torch = dep_node.accept(self)
        return torch.logsumexp(dep_torch, -1)
    @override
    @utils.instance_memo
    def visit_softmax_node(self, node):
        dep_node, = node.get_direct_dependencies()
        dep_torch = dep_node.accept(self)
        return torch.softmax(dep_torch, -1)
    @override
    @utils.instance_memo
    def visit_avg_node(self, node):
        dep_node, = node.get_direct_dependencies()
        dep_torch = dep_node.accept(self)
        return dep_torch.mean(tuple(range(node._n_axes_to_avg)))
    @override
    @utils.instance_memo
    def visit_cross_entropy_logits_node(self, node):
        yhat_logits_node, y_probas_node = node.get_direct_dependencies()
        yhat_logits_torch, y_probas_torch = yhat_logits_node.accept(self), y_probas_node.accept(self)
        return torch.nn.functional.cross_entropy(yhat_logits_torch, y_probas_torch, reduction="none")
