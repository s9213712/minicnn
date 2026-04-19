from __future__ import annotations

from minicnn.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params=list(params), lr=lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        updated = 0
        for p in self.params:
            if p.grad is None:
                continue
            try:
                grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
                p.data = p.data - self.lr * grad
                updated += 1
            except Exception:
                # Keep optimizer layer non-fatal for metadata-only parameters.
                continue
        return {'updated': updated, 'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay}
