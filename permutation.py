
from typing import *
import dataclasses

@dataclasses.dataclass
class Permutation:
    permutation: tuple[int, ...]

    def __post_init__(self):
        assert all(x >= 0 for x in self.permutation)
        assert all(x < len(self.permutation) for x in self.permutation)
    
    @staticmethod
    def bring_to_front(indices: tuple[int, ...]) -> "Permutation":
        """Return the permutation that brings to front the selected indices."""
        raise NotImplementedError() # TODO
    
    @staticmethod
    def bring_to_back(indices: tuple[int, ...]) -> "Permutation":
        """Return the permutation that brings to back the selected indices."""
        raise NotImplementedError() # TODO
