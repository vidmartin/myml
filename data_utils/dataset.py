
from typing import *
from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):
    """
    Represents a classic dataset, where each datapoint consists of two multidimensional arrays `X` and `Y`,
    with `X` holding the features and `Y` the target values.
    """
    @abstractmethod
    def slice_(self, start: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a batch of training data as a tuple `(X, Y)`, where `X` are the features and `Y` are targets,
        such that `(X[i,...], Y[i,...])` is the `(start + i)`-th datapoint in the dataset and
        `X.shape[0] == Y.shape[0] <= max_len`.
        """
        raise NotImplementedError()
    @abstractmethod
    def __len__(self) -> int:
        """Return number of datapoints in the dataset."""
        raise NotImplementedError()
