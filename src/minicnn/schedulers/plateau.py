from __future__ import annotations

from minicnn.schedulers.base import LRScheduler


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer, factor: float = 0.5, patience: int = 3, min_lr: float = 1e-5):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = None
        self.bad_epochs = 0

    def step(self, metric=None):
        self.last_epoch += 1
        if metric is None:
            return self.get_last_lr()
        if self.best is None or metric < self.best:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                self.optimizer.lr = max(self.min_lr, self.optimizer.lr * self.factor)
                self.bad_epochs = 0
        return self.get_last_lr()
