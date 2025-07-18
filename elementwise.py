
from typing import *
import dataclasses
import numpy as np

@dataclasses.dataclass
class ElementwiseOperation:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    dfn: Callable[[np.ndarray], np.ndarray]

    def __repr__(self):
        return self.name

_SIN = ElementwiseOperation("sin", lambda x: np.sin(x), lambda x: np.cos(x))
_COS = ElementwiseOperation("cos", lambda x: np.cos(x), lambda x: -np.sin(x))
_EXP = ElementwiseOperation("exp", lambda x: np.exp(x), lambda x: np.exp(x))
_LOG = ElementwiseOperation("log", lambda x: np.log(x), lambda x: 1 / x)
_RELU = ElementwiseOperation("relu", lambda x: x * (x > 0), lambda x: (x > 0).astype(x.dtype))

class CommonElementwiseOperations:
    @staticmethod
    def ipow(pow: int):
        return ElementwiseOperation(
            f"ipow({pow})",
            lambda x: x ** pow,
            lambda x: pow * x ** (pow - 1),
        )
    @staticmethod
    def mulby(num: float):
        return ElementwiseOperation(
            f"mulby({num})",
            lambda x: x * num,
            lambda x: np.full(x.shape, num),
        )
    @staticmethod
    def sin(): return _SIN
    @staticmethod
    def cos(): return _COS
    @staticmethod
    def exp(): return _EXP
    @staticmethod
    def log(): return _LOG
    @staticmethod
    def relu(): return _RELU
