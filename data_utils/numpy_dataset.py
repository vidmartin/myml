
from typing import *
import numpy as np
from data_utils.dataset import Dataset

class NumpyDataset(Dataset):
    def __init__(self, X_arr: np.ndarray, Y_arr: np.ndarray):
        assert self._X_arr.shape[0] == self._Y_arr.shape[0], \
            f"the array with features and the array with target values must have the same number of datapoints (i.e. equal size in the first dimension)"
        self._X_arr = X_arr
        self._Y_arr = Y_arr
    @override
    def slice_(self, start: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        end = min(start + max_len, self._X_arr.shape[0])
        return (self._X_arr[start:end,...], self._Y_arr[start:end,...])
    @override
    def __len__(self):
        return self._X_arr.shape[0]
