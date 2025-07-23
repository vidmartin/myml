
from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import functools
from warnings import deprecated
import numpy as np
import elementwise
import permutation
import utils
from node_visitor.node_visitor import NodeVisitor

TResult = TypeVar("TResult")

class TensorNode(ABC):
    @abstractmethod
    def get_value(self) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_direct_dependencies(self) -> list[TensorNode]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_leaf_dependencies(self) -> set[ConstantNode]:
        raise NotImplementedError()
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
    
    def get_gradients_against(self, leaves: list[ConstantNode], output_gradient: np.ndarray | None = None):
        results = [np.zeros(leaf.get_shape()) for leaf in leaves]
        inputs_grads = self.get_input_gradients(output_gradient)
        for input_grad, input_ in zip(inputs_grads, self.get_direct_dependencies()):
            grads = input_.get_gradients_against(leaves, input_grad)
            for i in range(len(grads)):
                # vaguely: according to chain rule, gradients add up
                # TODO: understand more formally
                results[i] += grads[i]
        return results
    
    @final
    def get_input_gradients(self, output_gradient: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Returns "local gradients" of inputs given the gradient of the output.
        """
        shape = self.get_shape()
        if output_gradient is None:
            assert functools.reduce(lambda a, b: a * b, shape, 1) == 1, "`output_gradient` is optional only for nodes with scalar outputs"
            output_gradient = np.ones(shape, dtype=np.float32)
        assert output_gradient.shape == self.get_shape(), "`output_gradient` must have the same shape as the output of the given node"
        return self._get_input_gradients(output_gradient)
    @abstractmethod
    def _get_input_gradients(self, output_gradient: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError()
    
    @abstractmethod
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        raise NotImplementedError()

class ConstantNode(TensorNode):
    def __init__(self, value: np.ndarray, name: str | None = None):
        self._value = value
        self._deps: list[TensorNode] = []
        self._ders: list[np.ndarray] = []
        self._name = name
    
    @override
    def get_value(self):
        return self._value
    @override
    def get_shape(self):
        return tuple(self._value.shape)
    @override
    def get_direct_dependencies(self):
        return self._deps
    @override
    def get_leaf_dependencies(self):
        return { self }
    @override
    def __repr__(self):
        return f"ConstantNode({"\"" + self._name + "\"" if self._name is not None else "<anon>"})"
    
    @override
    def get_gradients_against(self, leaves: list[ConstantNode], output_gradient: np.ndarray | None = None):
        return [
            output_gradient if leaf is self else np.zeros(leaf.get_shape())
            for leaf in leaves
        ]
    @override
    def _get_input_gradients(self, output_gradient):
        return []
    
    @override
    def accept(self, visitor):
        return visitor.visit_constant_node(self)
    
class LazyDependentNode(TensorNode):
    def __init__(self, deps: list[TensorNode]):
        self._value: np.ndarray | None = None
        self._deps = deps
        self._ders: list[np.ndarray] | None = None
        self._leaf_dependencies: set[ConstantNode] | None = None

    @override
    @final
    def get_value(self):
        if self._value is None:
            self._value = self._get_value()
        return self._value
    @abstractmethod
    def _get_value(self) -> np.ndarray:
        raise NotImplementedError()
    
    @override
    @final
    def get_direct_dependencies(self):
        return self._deps
    @override
    @final
    def get_leaf_dependencies(self):
        if self._leaf_dependencies is None:
            self._leaf_dependencies = self._get_leaf_dependencies()
        return self._leaf_dependencies
    def _get_leaf_dependencies(self):
        return set().union(*[
            dep.get_leaf_dependencies()
            for dep in self.get_direct_dependencies()
        ])
    
class ElementwiseNode(LazyDependentNode):
    def __init__(self, function: elementwise.ElementwiseFunction, deps: list[TensorNode]):
        super().__init__(deps)
        self._function = function

        shapes = {dep.get_shape() for dep in deps}
        assert len(shapes) == 1, f"cannot perform {function.get_input_count()}-ary elementwise function on tensors of differing shapes ({shapes})"
        # ^ we want to prevent broadcasting, that would require more careful calculation of derivatives ^
        # TODO: enable broadcasting
        self._shape: tuple[int, ...] = next(iter(shapes))
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        return self._function.evaluate_function([
            dep.get_value() for dep in self.get_direct_dependencies()
        ])
    @override
    def _get_input_gradients(self, output_gradient: np.ndarray):
        deps = self.get_direct_dependencies()
        dep_values = [dep.get_value() for dep in deps]
        return [
            output_gradient * self._function.evaluate_partial_derivative(i, dep_values)
            for i in range(len(deps))
        ]
    
    @override
    def __repr__(self):
        return f"ElementwiseNode({self._function}, [{', '.join(repr(dep) for dep in self.get_direct_dependencies())}])"
    
    @override
    def accept(self, visitor):
        return visitor.visit_elementwise_node(self)

class TensorDotNode(LazyDependentNode):
    def __init__(
        self,
        lhs: TensorNode,
        rhs: TensorNode,
        n_axes_to_contract: int,
    ):
        super().__init__([lhs, rhs])

        lhs_sh = lhs.get_shape()
        rhs_sh = rhs.get_shape()
        assert len(lhs_sh) >= n_axes_to_contract
        assert len(rhs_sh) >= n_axes_to_contract
        assert all(lhs_sh[-n_axes_to_contract + i] == rhs_sh[i] for i in range(n_axes_to_contract)), \
            f"the shapes of the inputs aren't compatible with TensorDotNode; lhs shape is {lhs_sh}, rhs shape is {rhs_sh}"

        self._lhs = lhs
        self._rhs = rhs
        self._n_axes_to_contract = n_axes_to_contract
        self._shape = lhs_sh[:-n_axes_to_contract] + rhs_sh[n_axes_to_contract:]
    
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self) -> np.ndarray:
        lhs_v = self._lhs.get_value()
        rhs_v = self._rhs.get_value()
        return np.tensordot(lhs_v, rhs_v, self._n_axes_to_contract)
    
    @override
    def _get_input_gradients(self, output_gradient):
        lhs_sh, rhs_sh = self._lhs.get_shape(), self._rhs.get_shape()
        n_lhs_free_axes, n_rhs_free_axes = \
            len(lhs_sh) - self._n_axes_to_contract, \
            len(rhs_sh) - self._n_axes_to_contract
        lhs_v, rhs_v = self._lhs.get_value(), self._rhs.get_value()
        return [
            np.tensordot(
                output_gradient,
                rhs_v.transpose(
                    tuple(range(self._n_axes_to_contract, len(rhs_sh))) + \
                    tuple(range(self._n_axes_to_contract))
                ),
                n_rhs_free_axes
            ),
            np.tensordot(
                lhs_v.transpose(
                    tuple(range(n_lhs_free_axes, len(lhs_sh))) + \
                    tuple(range(n_lhs_free_axes))
                ),
                output_gradient,
                n_lhs_free_axes
            )
        ]
    
    @override
    def __repr__(self):
        return f"TensorDotNode({self._lhs}, {self._rhs}, {self._n_axes_to_contract})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_tensor_dot_node(self)

class TransposeNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, permutation: permutation.Permutation):
        dep_sh = dep.get_shape()
        assert len(permutation.permutation) == len(dep_sh)
        super().__init__([dep])
        self._permutation = permutation
        self._inverse_permutation = permutation.inverse()
        self._shape = tuple(dep_sh[permutation.permutation[i]] for i in range(len(dep_sh)))
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        return self._deps[0].get_value().transpose(self._permutation.permutation)
    @override
    def _get_input_gradients(self, output_gradient):
        return [output_gradient.transpose(self._inverse_permutation.permutation)]
    @override
    def __repr__(self):
        return f"TransposeNode({self._deps[0]}, {self._permutation})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_transpose_node(self)

class ExtendNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, prepend_dims: tuple[int, ...]):
        super().__init__([dep])
        self._prepend_dims = prepend_dims
    @override
    def get_shape(self):
        return self._prepend_dims + self._deps[0].get_shape()
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        indexer = (np.newaxis,) * len(self._prepend_dims) + (slice(0, None),) * len(depval.shape)
        dummy = np.zeros(self._prepend_dims + (1,) * len(depval.shape))
        return depval[indexer] + dummy
    @override
    def _get_input_gradients(self, output_gradient):
        return [np.sum(output_gradient, tuple(range(len(self._prepend_dims))))]
    @override
    def __repr__(self):
        return f"ExtendNode({self._deps[0]}, {self._prepend_dims})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_extend_node(self)
    
class SumNode(LazyDependentNode):
    """
    Sum along the first `n_axes_to_sum` dimensions.
    
    Use together with `TransposeNode` to get sum along any dimension.
    """
    def __init__(self, dep: TensorNode, n_axes_to_sum: int):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        assert n_axes_to_sum <= len(dep_sh)
        self._shape = dep_sh[n_axes_to_sum:]
        self._n_axes_to_sum = n_axes_to_sum
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return depval.sum(tuple(range(self._n_axes_to_sum)))
    @override
    def _get_input_gradients(self, output_gradient):
        dep_sh = self._deps[0].get_shape()
        indexer = (np.newaxis,) * self._n_axes_to_sum + (slice(0, None),) * len(self.get_shape())
        return [output_gradient[indexer] + np.zeros(dep_sh)]
    @override
    def __repr__(self):
        return f"SumNode({self._deps[0]}, {self._n_axes_to_sum})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_sum_node(self)

class LogSumExpNode(LazyDependentNode):
    """
    Perform the LogSumExp operation along the last dimension.

    Use together with `TransposeNode` to perform LogSumExp along any dimension.
    """
    def __init__(self, dep: TensorNode):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        self._shape = dep_sh[:-1]
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return utils.log_sum_exp(depval)
    @override
    def _get_input_gradients(self, output_gradient):
        depval = self._deps[0].get_value()
        softmaxed = utils.softmax(depval)
        return [output_gradient[...,np.newaxis] * softmaxed]
    @override
    def __repr__(self):
        return f"LogSumExpNode({self._deps[0]})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_logsumexp_node(self)
    
    
class SoftmaxNode(LazyDependentNode):
    """
    Perform softmax over the last dimension.

    Use together with `TransposeNode` to perform softmax along any dimension.
    """
    def __init__(self, dep: TensorNode):
        super().__init__([dep])
    @override
    def get_shape(self):
        return self._deps[0].get_shape()
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        return utils.softmax(depval)
    @override
    def _get_input_gradients(self, output_gradient):
        depval = self._deps[0].get_value()
        softmaxed = utils.softmax(depval)
        temp = output_gradient * softmaxed
        return [temp - softmaxed * temp.sum(-1)[...,np.newaxis]]
    @override
    def __repr__(self):
        return f"SoftmaxNode({self._deps[0]})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_softmax_node(self)
    
class WrappingNode(LazyDependentNode):
    def __init__(self, deps: list[TensorNode], wrapper: Callable[[list[ConstantNode]], TensorNode]):
        super().__init__(deps)
        self._wrapper = wrapper

        self._initialized = False
        self._leaves: list[ConstantNode] | None = None
        self._wrapped: TensorNode | None = None
    def _initialize(self):
        assert not self._initialized
        self._leaves = [
            ConstantNode(dep.get_value())
            for dep in self.get_direct_dependencies()
        ]
        self._wrapped = self._wrapper(self._leaves)
        self._initialized = True
    @override
    def get_shape(self):
        if not self._initialized:
            self._initialize()
        assert self._wrapped is not None
        return self._wrapped.get_shape()
    @override
    def _get_value(self):
        if not self._initialized:
            self._initialize()
        assert self._wrapped is not None
        return self._wrapped.get_value()
    @override
    def _get_input_gradients(self, output_gradient):
        if not self._initialized:
            self._initialize()
        assert self._wrapped is not None
        assert self._leaves is not None
        return self._wrapped.get_gradients_against(self._leaves, output_gradient)
    @override
    def __repr__(self):
        if not self._initialized:
            return f"WrappingNode({self._deps}, <lambda>) {{ unitialized }}"
        return f"WrappingNode({self._deps}, <lambda>) {{ {self._wrapped} }}"
    
    @override
    def accept(self, visitor: NodeVisitor):
        raise NotImplementedError()
    
class AvgNode(WrappingNode):
    """
    Computes the average (aka mean) along the first dimension.

    Use together with `TransposeNode` to compute average along any dimension.
    """
    def __init__(self, dep: TensorNode, n_axes_to_avg: int):
        dep_shape = dep.get_shape()
        def construct(nodes: list[ConstantNode]):
            input_, = nodes
            sum_node = SumNode(input_, n_axes_to_avg)
            divisor = functools.reduce(lambda a, b: a * b, dep_shape[:n_axes_to_avg])
            scale_node = ElementwiseNode(elementwise.ElementwiseScale(1.0 / divisor), [sum_node])
            return scale_node
        self._n_axes_to_avg = n_axes_to_avg # not used here but may be used by visitors (we allow visitors to inspect private attrs)
        super().__init__([dep], construct)
    @override
    def __repr__(self):
        return f"AvgNode({self._deps[0]})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_avg_node(self)
    
class CrossEntropyLogitsNode(WrappingNode):
    """
    Calculate cross entropy loss along the last dimension, where `yhat_logits` are predicted *logits* and `y_probas` are the target *probabilities*.

    Use together with `TransposeNode` to perform CrossEntropyLogits along any dimension.
    """
    def __init__(self, yhat_logits: TensorNode, y_probas: TensorNode):
        dep_shape = yhat_logits.get_shape()
        assert dep_shape == y_probas.get_shape()
        self._yhat = yhat_logits
        self._y = y_probas
        self._shape = dep_shape[:-1]

        def construct(nodes: list[ConstantNode]):
            yhat_node, y_node = nodes

            mul_node = ElementwiseNode(elementwise.ElementwiseMul(2), [yhat_node, y_node])
            sum_node = SumNode(TransposeNode(mul_node, permutation.Permutation.bring_to_front((-1,), len(dep_shape))), 1)
            neg_sum_node = ElementwiseNode(elementwise.ElementwiseScale(-1.0), [sum_node])

            logsumexp_node = LogSumExpNode(yhat_node)
            sum_probas_node = SumNode(TransposeNode(y_node, permutation.Permutation.bring_to_front((-1,), len(dep_shape))), 1)
            logsumexp_mul_node = ElementwiseNode(elementwise.ElementwiseMul(2), [logsumexp_node, sum_probas_node])
            # ^ probabilities should add up to one (when used correctly), so the multiplication by sum_probas_node is redundant in forward pass
            # however, it is important for computation of gradient against y_node
            # however, in practical use cases one rarely needs to compute that gradient
            # CONSIDER: make the sum_probas_node & logsumexp_mul_node optional

            add_node = ElementwiseNode(elementwise.ElementwiseAdd(2), [neg_sum_node, logsumexp_mul_node])
            return add_node

        super().__init__([yhat_logits, y_probas], construct)
    @override
    def __repr__(self):
        return f"CrossEntropyLogitsNode({self._yhat}, {self._y})"
    
    @override
    def accept(self, visitor):
        return visitor.visit_cross_entropy_logits_node(self)
    
# TODO: FlattenNode
    