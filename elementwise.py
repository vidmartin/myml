
import functools
from typing import *
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from elementwise_visitor.elementwise_visitor import ElementwiseVisitor

class ElementwiseFunction(ABC):
    @abstractmethod
    def get_input_count(self):
        raise NotImplementedError()

    @final
    def evaluate_function(self, inputs: list[np.ndarray]) -> np.ndarray:
        assert len(inputs) == self.get_input_count()
        return self._evaluate_function(inputs)
    @final
    def evaluate_partial_derivative(self, input_index: int, inputs: list[np.ndarray]) -> np.ndarray:
        assert len(inputs) == self.get_input_count()
        assert input_index >= 0
        assert input_index < len(inputs)
        return self._evaluate_partial_derivative(input_index, inputs)

    @abstractmethod
    def _evaluate_function(self, inputs: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()
    @abstractmethod
    def _evaluate_partial_derivative(self, input_index: int, inputs: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def accept(self, visitor: ElementwiseVisitor):
        raise NotImplementedError()

@dataclasses.dataclass
class ElementwiseAdd(ElementwiseFunction):
    input_count: int
    def __post_init__(self):
        assert self.input_count > 0

    @override
    def get_input_count(self):
        return self.input_count
    @override
    def _evaluate_function(self, inputs):
        return sum(inputs)
    @override
    def _evaluate_partial_derivative(self, input_index, inputs):
        assert len(set(input_.shape for input_ in inputs)) == 1
        return np.ones(inputs[0].shape, dtype=np.float32)
    
    @override
    def accept(self, visitor):
        return visitor.visit_add(self)

@dataclasses.dataclass
class ElementwiseMul(ElementwiseFunction):
    input_count: int
    def __post_init__(self):
        assert self.input_count > 0

    @override
    def get_input_count(self):
        return self.input_count
    @override
    def _evaluate_function(self, inputs):
        return functools.reduce(lambda a, b: a * b, inputs)
    @override
    def _evaluate_partial_derivative(self, input_index, inputs):
        assert len(set(input_.shape for input_ in inputs)) == 1
        return functools.reduce(
            lambda a, b: a * b,
            [input_ for i, input_ in enumerate(inputs) if i != input_index]
        )
    
    @override
    def accept(self, visitor):
        return visitor.visit_mul(self)
    
class ElementwiseUnary(ElementwiseFunction):
    @override
    def get_input_count(self):
        return 1
    
    @override
    @final
    def _evaluate_function(self, inputs):
        return self._evaluate_unary_function(inputs[0])
    @override
    @final
    def _evaluate_partial_derivative(self, input_index, inputs):
        return self._evaluate_unary_derivative(inputs[0])

    @abstractmethod
    def _evaluate_unary_function(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    @abstractmethod
    def _evaluate_unary_derivative(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

@dataclasses.dataclass
class ElementwiseSin(ElementwiseUnary):
    @override
    def _evaluate_unary_function(self, input):
        return np.sin(input)
    @override
    def _evaluate_unary_derivative(self, input):
        return np.cos(input)
    
    @override
    def accept(self, visitor):
        return visitor.visit_sin(self)

@dataclasses.dataclass    
class ElementwiseCos(ElementwiseUnary):
    @override
    def _evaluate_unary_function(self, input):
        return np.cos(input)
    @override
    def _evaluate_unary_derivative(self, input):
        return -np.sin(input)
    
    @override
    def accept(self, visitor):
        return visitor.visit_cos(self)

@dataclasses.dataclass
class ElementwiseIPow(ElementwiseUnary):
    power: int

    @override
    def _evaluate_unary_function(self, input):
        return input ** self.power
    @override
    def _evaluate_unary_derivative(self, input):
        return self.power * input ** (self.power - 1)
    
    @override
    def accept(self, visitor):
        return visitor.visit_ipow(self)
    
@dataclasses.dataclass
class ElementwiseFPow(ElementwiseUnary):
    power: float

    @override
    def _evaluate_unary_function(self, input):
        return input ** self.power
    @override
    def _evaluate_unary_derivative(self, input):
        return self.power * input ** (self.power - 1)
    
    @override
    def accept(self, visitor):
        return visitor.visit_fpow(self)
    
# CONSIDER: merge IPow & FPow into one, it's annoying like this and benefit is questionable
    
@dataclasses.dataclass
class ElementwiseAbs(ElementwiseUnary):
    @override
    def _evaluate_unary_function(self, input):
        return np.abs(input)
    @override
    def _evaluate_unary_derivative(self, input):
        return np.sign(input)
    
    @override
    def accept(self, visitor):
        return visitor.visit_abs(self)

@dataclasses.dataclass
class ElementwiseExp(ElementwiseUnary):
    @override
    def _evaluate_unary_function(self, input):
        return np.exp(input)
    @override
    def _evaluate_unary_derivative(self, input):
        return np.exp(input)

    @override
    def accept(self, visitor):
        return visitor.visit_exp(self)

@dataclasses.dataclass
class ElementwiseLog(ElementwiseUnary):
    @override
    def _evaluate_unary_function(self, input):
        return np.log(input)
    @override
    def _evaluate_unary_derivative(self, input):
        return 1.0 / input
    
    @override
    def accept(self, visitor):
        return visitor.visit_log(self)

@dataclasses.dataclass
class ElementwiseScale(ElementwiseUnary):
    factor: float

    @override
    def _evaluate_unary_function(self, input):
        return input * self.factor
    @override
    def _evaluate_unary_derivative(self, input):
        return np.full(input.shape, self.factor)
    
    @override
    def accept(self, visitor):
        return visitor.visit_scale(self)

@dataclasses.dataclass
class ElementwiseReLU(ElementwiseUnary):
    @override
    def _evaluate_unary_function(self, input):
        return input * (input > 0)
    @override
    def _evaluate_unary_derivative(self, input):
        return (input > 0).astype(input.dtype)
    
    @override
    def accept(self, visitor):
        return visitor.visit_relu(self)

@dataclasses.dataclass
class ElementwiseCrossLog(ElementwiseFunction):
    @override
    def get_input_count(self):
        return 2
    @override
    def _evaluate_function(self, inputs):
        lhs, rhs = inputs
        res = lhs * np.log(rhs)
        res[(lhs == 0.0) & (rhs == 0.0)] = 0.0
        return res
    @override
    def _evaluate_partial_derivative(self, input_index, inputs):
        lhs, rhs = inputs
        if input_index == 0:
            return np.log(rhs)
        elif input_index == 1:
            res = lhs / rhs
            res[(lhs == 0.0) & (rhs == 0.0)] = 0.0
            return res
        
    @override
    def accept(self, visitor):
        return visitor.visit_cross_log(self)

@dataclasses.dataclass
class ElementwiseSquaredDifference(ElementwiseFunction):
    @override
    def get_input_count(self):
        return 2
    @override
    def _evaluate_function(self, inputs):
        lhs, rhs = inputs
        return (lhs - rhs) ** 2
    @override
    def _evaluate_partial_derivative(self, input_index, inputs):
        lhs, rhs = inputs
        return 2 * (lhs - rhs) * (-1)**input_index
    
    @override
    def accept(self, visitor):
        return visitor.visit_squared_difference(self)
