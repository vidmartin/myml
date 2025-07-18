
from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import functools
from warnings import deprecated
import numpy as np
import elementwise

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
    @deprecated("deprecated - very inefficient")
    def get_derivatives(self) -> list[np.ndarray]:
        """
        Each item in the resulting list corredponds to one of the direct dependencies, in the same order as given by self.get_direct_dependencies().

        Denote the result of this node C and consider a dependency A. Denote the item corresponding to A in the resulting list as D.

        Then, if `C` is of shape `(n_1, ..., n_N)` and A is of shape `(m_1, ..., m_M)`, `D` will be of shape `(n_1, ..., n_N, m_1, ..., m_M)` and
        it will hold that `D_{i_1,...,i_N,j_1,...,j_M} = ∂C_{i_1,...,i_N} / ∂A_{j_1,...,j_M}`.

        This is a generalization of the Jacobian matrix for functions of multi-dimensional arrays. However it turns out that the result is very sparse
        (contains a lot of zeros) which makes gradient computation using this very inefficient.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_leaf_dependencies(self) -> set[ConstantNode]:
        raise NotImplementedError()
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
    
    @deprecated("inefficient calculation using the deprecated self.get_derivatives()")
    def get_gradients_against_old(self, leaves: list[ConstantNode]) -> list[np.ndarray]:
        leaf_deps = self.get_leaf_dependencies()
        leaf_dep_indices = [i for i, leaf in enumerate(leaves) if leaf in leaf_deps]
        gradients = [np.zeros(self.get_shape() + leaf.get_shape()) for leaf in leaves]
        leaves_of_interest = [leaves[i] for i in leaf_dep_indices]
        for dep, der in zip(self.get_direct_dependencies(), self.get_derivatives()):
            temp_gradients = dep.get_gradients_against_old(leaves_of_interest)
            for i, temp_grad in zip(leaf_dep_indices, temp_gradients):
                if leaves[i] is dep:
                    gradients[i] += der
                else:
                    gradients[i] += np.tensordot(der, temp_grad, len(dep.get_shape()))
        return gradients
    
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
    def get_derivatives(self):
        return self._ders
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
            if np.isnan(self._value).any():
                # TODO: delete this and similar code
                print(f"nan detected by {self}")
        return self._value
    @abstractmethod
    def _get_value(self) -> np.ndarray:
        raise NotImplementedError()
    
    @override
    @final
    def get_derivatives(self):
        if self._ders is None:
            self._ders = self._get_derivatives()
        return self._ders
    @abstractmethod
    def _get_derivatives(self) -> list[np.ndarray]:
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
    def _get_derivatives(self):
        raise NotImplementedError()
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

# class ElementwiseAddNode(LazyDependentNode):
#     def __init__(self, deps: list[TensorNode]):
#         super().__init__(deps)

#         shapes = {dep.get_shape() for dep in deps}
#         assert len(shapes) == 1, f"cannot elementwise-multiply tensors of differing shapes {shapes}"
#         # ^ we want to prevent broadcasting, that would require more careful calculation of derivatives ^
#         # TODO: enable broadcasting
#         self._shape: tuple[int, ...] = next(iter(shapes))

#     @override
#     def get_shape(self):
#         return self._shape

#     @override
#     def _get_value(self) -> np.ndarray:
#         deps = self.get_direct_dependencies()
#         assert deps
#         depvals = [dep.get_value() for dep in deps]
#         return sum(depvals)

#     @override
#     def _get_derivatives(self) -> list[np.ndarray]:
#         ders: list[np.ndarray] = []
#         for _ in self.get_direct_dependencies():
#             # let O be this node's output tensor and A the current dependency
#             # suppose O and A both have shape (n_1, \cdots, n_N)
#             # we define the derivative of O with respect to A, let's denote it ∂O/∂A, as
#             # a tensor with shape (n_1, \cdots, n_N, n_1, \cdots n_N) such that
#             # (∂O/∂A)_{i_1,\cdots,i_N,j_1,\cdots,j_N} = ∂O_{i_1,\cdots,i_N}/∂A_{j_1,\cdots,j_N}
#             # in this case we have ∂O_{i_1,\cdots,i_N}/∂A_{j_1,\cdots,j_N} = δ_{i_1,j_1} \cdot \cdots \cdot δ_{i_N,j_N}

#             indices = np.indices(self._shape * 2)
#             ders.append(
#                 functools.reduce(
#                     lambda a, b: a & b,
#                     [indices[i] == indices[i + len(self._shape)] for i in range(len(self._shape))],
#                     np.ones((), dtype=bool)
#                 ).astype(np.float32)
#             )
#         return ders
    
#     @override
#     def _get_gradients_against(self, leaves, output_gradient):
#         return super()._get_gradients_against(leaves, output_gradient)
    
#     @override
#     def __repr__(self):
#         return f"ElementwiseAddNode([{', '.join(repr(node) for node in self._deps)}])"
    
# class ElementwiseMulNode(LazyDependentNode):
#     def __init__(self, deps: list[TensorNode]):
#         super().__init__(deps)

#         shapes = {dep.get_shape() for dep in deps}
#         assert len(shapes) == 1, f"cannot elementwise-multiply tensors of differing shapes {shapes}"
#         # ^ we want to prevent broadcasting, that would require more careful calculation of derivatives ^
#         # TODO: enable broadcasting
#         self._shape: tuple[int, ...] = next(iter(shapes))

#     @override
#     def get_shape(self):
#         return self._shape

#     @override
#     def _get_value(self) -> np.ndarray:
#         deps = self.get_direct_dependencies()
#         assert deps
#         depvals = [dep.get_value() for dep in deps]
#         return functools.reduce(lambda x, y: x * y, depvals, np.ones((), dtype=bool))

#     @override
#     def _get_derivatives(self) -> list[np.ndarray]:
#         ders: list[np.ndarray] = []
#         deps = self.get_direct_dependencies()
#         depvals = [dep.get_value() for dep in deps]
#         for i in range(len(deps)):
#             # let O be this node's output tensor and A the current dependency
#             # suppose O and A both have shape (n_1, \cdots, n_N)
#             # we define the derivative of O with respect to A, let's denote it ∂O/∂A, as
#             # a tensor with shape (n_1, \cdots, n_N, n_1, \cdots n_N) such that
#             # (∂O/∂A)_{i_1,\cdots,i_N,j_1,\cdots,j_N} = ∂O_{i_1,\cdots,i_N}/∂A_{j_1,\cdots,j_N}
#             # in this case we have ∂O_{i_1,\cdots,i_N}/∂A_{j_1,\cdots,j_N} = P \cdot δ_{i_1,j_1} \cdot \cdots \cdot δ_{i_N,j_N},
#             # where P is the product of the corresponding values of the other dependencies -- e.g. if all the dependencies are A,B,C,D,
#             # we have P = B_{i_1,\cdots,i_N} \cdot C_{i_1,\cdots,i_N} \cdot D_{i_1,\cdots,i_N}

#             indices = np.indices(self._shape * 2)
#             product: np.ndarray = functools.reduce(
#                 lambda a, b: a * b,
#                 [depval for j, depval in enumerate(depvals) if j != i],
#                 np.ones((), dtype=bool)
#             )
#             indexer = (slice(0, None),) * len(self._shape) + (np.newaxis,) * len(self._shape)
#             product = product[indexer]

#             ders.append(
#                 product * functools.reduce(
#                     lambda a, b: a & b,
#                     [indices[i] == indices[i + len(self._shape)] for i in range(len(self._shape))],
#                     np.ones((), dtype=bool)
#                 ).astype(np.float32)
#             )
#         return ders
    
#     @override
#     def __repr__(self):
#         return f"ElementwiseMulNode([{', '.join(repr(node) for node in self._deps)}])"
    

# class ElementwiseUnaryNode(LazyDependentNode):
#     def __init__(
#         self,
#         dep: TensorNode,
#         op: elementwise.ElementwiseOperation,
#     ):
#         super().__init__([dep])
#         self._op = op
    
#     @override
#     def get_shape(self):
#         return self._deps[0].get_shape()
#     @override
#     def get_direct_dependencies(self):
#         return self._deps
#     @override
#     def _get_value(self):
#         return self._op.fn(self._deps[0].get_value())
#     @override
#     def _get_derivatives(self) -> list[np.ndarray]:
#         shape = self.get_shape()
#         indices = np.indices(shape * 2)
#         elementwise_der: np.ndarray = self._op.dfn(self._deps[0].get_value())
#         indexer = (slice(0, None),) * len(shape) + (np.newaxis,) * len(shape)
#         elementwise_der = elementwise_der[indexer]

#         der = elementwise_der * functools.reduce(
#             lambda a, b: a & b,
#             [indices[i] == indices[i + len(shape)] for i in range(len(shape))],
#             np.ones((), dtype=bool)
#         ).astype(np.float32)
#         return [der]
#     @override
#     def __repr__(self):
#         return f"ElementwiseUnaryNode({self._deps[0]}, {self._op})"
    

# class ElementwiseCrossLogNode(LazyDependentNode):
#     """Returns lhs * log(rhs) with 0 * log(0) defined as 0"""
#     def __init__(self, lhs: TensorNode, rhs: TensorNode) -> None:
#         super().__init__([lhs, rhs])
#         self._lhs = lhs
#         self._rhs = rhs

#         shapes = {lhs.get_shape(), rhs.get_shape()}
#         assert len(shapes) == 1, f"cannot elementwise-multiply tensors of differing shapes {shapes}"
#         # ^ we want to prevent broadcasting, that would require more careful calculation of derivatives ^
#         # TODO: enable broadcasting
#         self._shape: tuple[int, ...] = next(iter(shapes))

#     @override
#     def get_shape(self):
#         return self._shape
    
#     @override
#     def _get_value(self):
#         lhs_v = self._lhs.get_value()
#         rhs_v = self._rhs.get_value()
#         if np.isnan(lhs_v).any() or np.isnan(rhs_v).any():
#             print("nan entering")
#         res = lhs_v * np.log(rhs_v)
#         mask = (lhs_v == 0.0) & (rhs_v == 0.0)
#         res[mask] = 0.0
#         if np.isnan(res).any():
#             print("nan exiting")
#         return res
    
#     @override
#     def _get_derivatives(self):
#         lhs_v = self._lhs.get_value()
#         rhs_v = self._rhs.get_value()
#         indices = np.indices(self._shape * 2)
#         rhs_interleaved = rhs_v[(np.newaxis,) * len(self._shape) + (slice(0, None),) * len(self._shape)]
#         lhs_der = rhs_interleaved * functools.reduce(
#             lambda x, y: x & y,
#             [indices[i] == indices[len(self._shape) + i] for i in range(len(self._shape))],
#             np.ones((), dtype=bool),
#         )

#         rhs_der = lhs_v / rhs_v
#         rhs_der[rhs_v == 0] = np.inf
#         rhs_der = rhs_der[(np.newaxis,) * len(self._shape) + (slice(0, None),) * len(self._shape)]
#         rhs_der = rhs_der * functools.reduce(
#             lambda x, y: x & y,
#             [indices[i] == indices[len(self._shape) + i] for i in range(len(self._shape))],
#             np.ones((), dtype=bool),
#         )

#         return [lhs_der, rhs_der]
    
#     @override
#     def __repr__(self):
#         return f"ElementwiseCrossLogNode({self._lhs}, {self._rhs})"

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
        assert all(lhs_sh[-n_axes_to_contract + i] == rhs_sh[i] for i in range(n_axes_to_contract))

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
    def _get_derivatives(self):
        shape = self.get_shape()
        lhs_sh, rhs_sh = self._lhs.get_shape(), self._rhs.get_shape()
        n_lhs_free_axes, n_rhs_free_axes = \
            len(lhs_sh) - self._n_axes_to_contract, \
            len(rhs_sh) - self._n_axes_to_contract
        indices_lhs = np.indices(shape + lhs_sh)
        indices_rhs = np.indices(shape + rhs_sh)

        rhs_v = self._rhs.get_value()
        rhs_v_transposed = rhs_v.transpose(
            tuple(range(self._n_axes_to_contract, len(rhs_sh))) + \
            tuple(range(self._n_axes_to_contract))
        )
        rhs_v_indexer = \
            (np.newaxis,) * n_lhs_free_axes + \
            (slice(0, None),) * n_rhs_free_axes + \
            (np.newaxis,) * n_lhs_free_axes + \
            (slice(0, None),) * self._n_axes_to_contract
        rhs_v_interleaved = rhs_v_transposed[rhs_v_indexer]

        lhs_v = self._lhs.get_value()
        lhs_v_indexer = \
            (slice(0, None),) * n_lhs_free_axes + \
            (np.newaxis,) * n_rhs_free_axes + \
            (slice(0, None),) * self._n_axes_to_contract + \
            (np.newaxis,) * n_rhs_free_axes
        lhs_v_interleaved = lhs_v[lhs_v_indexer]

        return [
            # derivative with respect to lhs contains elements from rhs:
            rhs_v_interleaved * (
                functools.reduce(
                    lambda a, b: a & b,
                    [
                        indices_lhs[i] == indices_lhs[i + len(shape)]
                        for i in range(n_lhs_free_axes)
                    ],
                    np.ones((), dtype=bool)
                ).astype(np.float32)
            ),

            # derivative with respect to rhs contains elements from lhs:
            lhs_v_interleaved * (
                functools.reduce(
                    lambda a, b: a & b,
                    [
                        indices_rhs[n_lhs_free_axes + i] == indices_rhs[len(shape) + self._n_axes_to_contract + i]
                        for i in range(n_rhs_free_axes)
                    ],
                    np.ones((), dtype=bool)
                ).astype(np.float32)
            )
        ]
    
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

class TransposeNode(LazyDependentNode):
    def __init__(self, dep: TensorNode, permutation: tuple[int, ...]):
        dep_sh = dep.get_shape()
        assert len(permutation) == len(dep_sh)
        super().__init__([dep])
        self._permutation = permutation
        inverse_permutation = [0] * len(permutation)
        for i in range(len(permutation)):
            inverse_permutation[permutation[i]] = i
        self._inverse_permutation = inverse_permutation
        self._shape = tuple(dep_sh[permutation[i]] for i in range(len(dep_sh)))
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        return self._deps[0].get_value().transpose(self._permutation)
    @override
    def _get_derivatives(self):
        sh = self.get_shape()
        dep_sh = self._deps[0].get_shape()
        indices = np.indices(sh + dep_sh)
        der = functools.reduce(
            lambda a, b: a & b,
            [
                indices[self._inverse_permutation[i]] == indices[len(sh) + i]
                for i in range(len(sh))
            ],
            np.ones((), dtype=bool)
        )
        return [der]
    @override
    def _get_input_gradients(self, output_gradient):
        return [output_gradient.transpose(self._inverse_permutation)]
    @override
    def __repr__(self):
        return f"TransposeNode({self._deps[0]}, {self._permutation})"

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
    def _get_derivatives(self):
        dep_shape = self._deps[0].get_shape()
        indices = np.indices(self._prepend_dims + 2 * dep_shape)
        der = functools.reduce(
            lambda a, b: a & b,
            [
                indices[len(self._prepend_dims) + i] == indices[len(self._prepend_dims) + len(dep_shape) + i]
                for i in range(len(dep_shape))
            ],
            np.ones((), dtype=bool)
        ).astype(np.float32)
        return [der]
    @override
    def _get_input_gradients(self, output_gradient):
        return [np.sum(output_gradient, tuple(range(len(self._prepend_dims))))]
    @override
    def __repr__(self):
        return f"ExtendNode({self._deps[0]}, {self._prepend_dims})"
    
def softmax_node(dep: TensorNode):
    """Softmax over the last axis."""
    # TODO: change into node (problem: _get_derivatives implementation)

    dep_sh = dep.get_shape()
    exp = ElementwiseNode(elementwise.ElementwiseExp(), [dep])
    # print(f"exp shape: {exp.get_shape()}")
    rowsum = TensorDotNode(exp, ConstantNode(np.ones(dep_sh[-1])), 1)
    # print(f"rowsum shape: {rowsum.get_shape()}")
    rowsum_expanded = TransposeNode(ExtendNode(rowsum, (dep_sh[-1],)), tuple(range(len(dep_sh)))[1:] + (0,))
    # print(f"rowsum_expanded shape: {rowsum_expanded.get_shape()}")
    rowsum_expanded_inverted = ElementwiseNode(elementwise.ElementwisePow(-1), [rowsum_expanded])
    # print(f"rowsum_expanded_inverted shape: {rowsum_expanded_inverted.get_shape()}")
    softmaxed = ElementwiseNode(elementwise.ElementwiseMul(2), [exp, rowsum_expanded_inverted])
    # print(f"softmaxed shape: {softmaxed.get_shape()}")
    return softmaxed

def cross_entropy_node(dep: TensorNode, target_distributions: np.ndarray):
    """Cross entropy over the last axis."""
    # TODO: take logits instead of probas as input, use log-sum-exp trick
    assert dep.get_shape() == target_distributions.shape
    shape = dep.get_shape()
    target_distributions_node = ConstantNode(target_distributions)
    mul_node = ElementwiseNode(elementwise.ElementwiseCrossLog(), [target_distributions_node, dep])
    rowsum = TensorDotNode(mul_node, ConstantNode(np.ones(shape[-1])), 1)
    neg_node = ElementwiseNode(elementwise.ElementwiseScale(-1.0), [rowsum])
    return neg_node

def sum_node(dep: TensorNode, axis: int):
    dep_sh = dep.get_shape()
    assert axis == 0, f"this provisional implementation can only sum over the 0th axis"
    ones = ConstantNode(np.ones(dep_sh[axis]))
    return TensorDotNode(ones, dep, 1)

def avg_node(dep: TensorNode, axis: int):
    sum_nd = sum_node(dep, axis)
    return ElementwiseNode(elementwise.ElementwiseScale(1.0 / dep.get_shape()[axis]), [sum_nd])

def mse_node(dep: TensorNode, target: np.ndarray):
    neg_target_node = ConstantNode(-target)
    add_node = ElementwiseNode(elementwise.ElementwiseAdd(2), [dep, neg_target_node])
    sq_node = ElementwiseNode(elementwise.ElementwisePow(2), [add_node])
    return avg_node(sq_node, 0)
