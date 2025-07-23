
from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import elementwise

TReturn = TypeVar("TReturn")

class ElementwiseVisitor(ABC, Generic[TReturn]):
    @abstractmethod
    def visit_add(self, obj: elementwise.ElementwiseAdd) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_mul(self, obj: elementwise.ElementwiseMul) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_sin(self, obj: elementwise.ElementwiseSin) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_cos(self, obj: elementwise.ElementwiseCos) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_pow(self, obj: elementwise.ElementwisePow) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_abs(self, obj: elementwise.ElementwiseAbs) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_exp(self, obj: elementwise.ElementwiseExp) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_log(self, obj: elementwise.ElementwiseLog) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_scale(self, obj: elementwise.ElementwiseScale) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_relu(self, obj: elementwise.ElementwiseReLU) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_cross_log(self, obj: elementwise.ElementwiseCrossLog) -> TReturn:
        raise NotImplementedError()
    @abstractmethod
    def visit_squared_difference(self, obj: elementwise.ElementwiseSquaredDifference) -> TReturn:
        raise NotImplementedError()
