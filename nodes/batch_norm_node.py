
from typing import *
import functools
import numpy as np
from node_visitor.node_visitor import NodeVisitor
from nodes.tensor_node import TensorNode
from nodes.lazy_dependent_node import LazyDependentNode
import utils # TODO: remove

TResult = TypeVar("TResult")

class BatchNormNode(LazyDependentNode):
    """
    Apply batch normalization over the first `n_axes` dimensions. This means that the
    first `n_axes` dimensions are considered to be the spatial dimensions, while the other
    dimensions are the channel dimensions (typically only 1). Use together with `TransposeNode`
    to get finer control over which dimensions are the channel dimensions.

    This node by itself *doesn't* apply the affine transform that's usually applied when performing
    batch normalization. So it only performs the standardization.
    """
    def __init__(self, dep: TensorNode, n_axes: int, delta: float = 1e-5):
        super().__init__([dep])
        dep_sh = dep.get_shape()
        assert n_axes <= len(dep_sh)
        self._n_axes = n_axes
        self._shape = dep_sh
        self._delta = delta
        self._mu_ext: np.ndarray | None = None
        self._sigsq_adj_ext: np.ndarray | None = None
    @override
    def get_shape(self):
        return self._shape
    @override
    def _get_value(self):
        depval = self._deps[0].get_value()
        extend_indexer = (np.newaxis,) * self._n_axes + (...,)

        mu = depval.mean(tuple(range(self._n_axes)))
        self._mu_ext = mu[extend_indexer]

        sigsq_adj = ((depval - self._mu_ext) ** 2).mean(tuple(range(self._n_axes)))
        sigsq_adj += self._delta
        self._sigsq_adj_ext = sigsq_adj[extend_indexer]

        return (depval - self._mu_ext) / (self._sigsq_adj_ext ** 0.5)
    @override
    def _get_input_gradients(self, output_gradient: np.ndarray):
        _ = self.get_value()
        depval = self._deps[0].get_value()
        extend_indexer = (np.newaxis,) * self._n_axes + (...,)
        mstar = functools.reduce(lambda x, y: x * y, self.get_shape()[:self._n_axes])

        temp1 = output_gradient - output_gradient.sum(tuple(range(self._n_axes)))[extend_indexer] / mstar
        temp2 = (depval - self._mu_ext) / mstar
        temp3 = (output_gradient * (depval - self._mu_ext)).sum(tuple(range(self._n_axes)))[extend_indexer]
        sigsq_adj_ext_sqrt = self._sigsq_adj_ext ** 0.5

        temp4 = sigsq_adj_ext_sqrt * temp1 - temp2 * temp3 / sigsq_adj_ext_sqrt
        return [temp4 / self._sigsq_adj_ext]
    @override
    def __repr__(self):
        return f"BatchNormNode({self._deps[0]}, {self._n_axes}, {self._delta})"
    
    @override
    def accept(self, visitor: NodeVisitor[TResult]) -> TResult:
        raise NotImplementedError()