
from typing import *
import numpy as np
from data_utils.dataset import Dataset
from data_utils.dataloader import Dataloader

class DatasetDataloader(Dataloader):
    def __init__(self, dataset: Dataset, batch_size: int):
        self._dataset = dataset
        self._batch_size = batch_size
    @override
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            start = i * self._batch_size
            yield self._dataset.slice_(start, self._batch_size)
    @override
    def __len__(self) -> int:
        return (len(self._dataset) + self._batch_size - 1) // self._batch_size
