from __future__ import annotations

from minicnn.optim.optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params=list(params), lr=lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        updated = 0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            if self.momentum:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                p.data = p.data + self.velocities[i]
            else:
                p.data = p.data - self.lr * grad
            updated += 1
        return {'updated': updated, 'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay}
