
from typing import *
from abc import ABC, abstractmethod
import numpy as np

T = TypeVar("T")
class Metric(ABC, Generic[T]):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()
    @abstractmethod
    def init_vars(self) -> T:
        raise NotImplementedError()
    @abstractmethod
    def process_batch(self, vars: T, features: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> T:
        raise NotImplementedError()
    @abstractmethod
    def get_result(self, vars: T) -> float:
        raise NotImplementedError()
    def format_result(self, result: float) -> str:
        return f"{result:.3f}"
