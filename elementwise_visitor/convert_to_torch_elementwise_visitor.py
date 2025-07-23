
from typing import *
from elementwise_visitor.elementwise_visitor import ElementwiseVisitor
import elementwise
import functools
import torch

class ConvertToTorchElementwiseVisitor(ElementwiseVisitor):
    """
    Applies our elementwise functions to PyTorch tensors instead of numpy arrays.
    
    Is here only for purposes of testing our framework against PyTorch. Requires the PyTorch optional dependency.
    """

    def __init__(self, deps: list[torch.Tensor]):
        self._deps = deps
    @override
    def visit_add(self, obj):
        return functools.reduce(lambda x, y: x + y, self._deps)
    @override
    def visit_mul(self, obj):
        return functools.reduce(lambda x, y: x * y, self._deps)
    @override
    def visit_sin(self, obj):
        dep, = self._deps
        return torch.sin(dep)
    @override
    def visit_cos(self, obj):
        dep, = self._deps
        return torch.cos(dep)
    @override
    def visit_pow(self, obj):
        dep, = self._deps
        return dep ** obj.power
    @override
    def visit_abs(self, obj):
        dep, = self._deps
        return torch.abs(dep)
    @override
    def visit_exp(self, obj):
        dep, = self._deps
        return torch.exp(dep)
    @override
    def visit_log(self, obj):
        dep, = self._deps
        return torch.log(dep)
    @override
    def visit_scale(self, obj):
        dep, = self._deps
        return dep * obj.factor
    @override
    def visit_relu(self, obj):
        dep, = self._deps
        return torch.relu(dep)
    @override
    def visit_cross_log(self, obj):
        lhs, rhs = self._deps
        res = lhs * torch.log(rhs)
        res[(lhs == 0.0) & (rhs == 0.0)] = 0.0
        return res
    @override
    def visit_squared_difference(self, obj):
        lhs, rhs = self._deps
        return (lhs - rhs) ** 2
