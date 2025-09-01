
from typing import *
from abc import ABC, abstractmethod
import numpy as np

class Dataloader(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns an iterator over the batches."""
        raise NotImplementedError()
    @abstractmethod
    def __len__(self) -> int:
        """Number of batches each iterator returned by this dataloader produces."""
        raise NotImplementedError()