
from typing import *
from data_utils.dataloader import Dataloader

class TakeDataloader(Dataloader):
    def __init__(self, wrapped: Dataloader, take: int):
        self._wrapped = wrapped
        self._take = take
    @override
    def __iter__(self):
        for i, batch in enumerate(self._wrapped):
            if i >= self._take:
                break
            yield batch
    @override
    def __len__(self):
        return min(self._take, len(self._wrapped))
