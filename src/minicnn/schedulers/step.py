from __future__ import annotations

from minicnn.schedulers.base import LRScheduler


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, min_lr: float = 0.0):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, metric=None):
        self.last_epoch += 1
        if self.last_epoch >= 0 and (self.last_epoch + 1) % self.step_size == 0:
            self.optimizer.lr = max(self.min_lr, self.optimizer.lr * self.gamma)
        self._last_lr = [self.optimizer.lr]
        return self.get_last_lr()
