from __future__ import annotations

import math

from minicnn.schedulers.base import LRScheduler


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, lr_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.lr_min = lr_min
        self._lr_max = optimizer.lr

    def step(self, metric=None):
        self.last_epoch += 1
        epoch = self.last_epoch + 1
        if self.T_max == 0:
            lr = self.lr_min
        else:
            lr = self.lr_min + 0.5 * (self._lr_max - self.lr_min) * (1.0 + math.cos(math.pi * epoch / self.T_max))
        self.optimizer.lr = lr
        self._last_lr = [lr]
        return self.get_last_lr()
