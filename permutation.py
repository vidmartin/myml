
from __future__ import annotations
from typing import *
import dataclasses

def normalize_indices(indices: tuple[int, ...], n_axes: int):
    return tuple(
        k if k >= 0 else k + n_axes
        for k in indices
    )

@dataclasses.dataclass(frozen=True, eq=True)
class Permutation:
    permutation: tuple[int, ...]

    def __post_init__(self):
        assert all(x >= 0 for x in self.permutation)
        assert all(x < len(self.permutation) for x in self.permutation)
    
    def inverse(self) -> Permutation:
        inverse_permutation = [0] * len(self.permutation)
        for i in range(len(self.permutation)):
            inverse_permutation[self.permutation[i]] = i
        return Permutation(inverse_permutation)
    
    def compose(self, other: Permutation) -> Permutation:
        assert len(self.permutation) == len(other.permutation)
        return Permutation(tuple(
            self.permutation[other.permutation[i]]
            for i in range(len(self.permutation))
        ))
    
    @staticmethod
    def create(permutation: tuple[int, ...]):
        """
        Creates a new permutation object.
        
        Unlike calling the constructor directly, this performs some convenient things
        like allowing negative integers to refer to the indices from the back.
        """
        return Permutation(normalize_indices(permutation, len(permutation)))

    @staticmethod
    def identity(n_indices: int) -> Permutation:
        """Return the identity permutation."""
        return Permutation(tuple(range(n_indices)))
    
    @staticmethod
    def bring_to_front(selected: tuple[int, ...], n_indices: int) -> Permutation:
        """Return the permutation that brings to front the selected indices."""
        selected = normalize_indices(selected, n_indices)
        return Permutation(selected + tuple(i for i in range(n_indices) if i not in selected))
    
    @staticmethod
    def bring_to_back(selected: tuple[int, ...], n_indices: int) -> Permutation:
        """Return the permutation that brings to back the selected indices."""
        selected = normalize_indices(selected, n_indices)
        return Permutation(tuple(i for i in range(n_indices) if i not in selected) + selected)
