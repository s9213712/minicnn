from __future__ import annotations


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = -1
        self._last_lr = [self.optimizer.lr]

    def step(self, metric=None):
        self._last_lr = [self.optimizer.lr]
        self.last_epoch += 1
        return self.get_last_lr()

    def get_last_lr(self):
        return list(self._last_lr)
