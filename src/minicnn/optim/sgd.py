from __future__ import annotations

import numpy as np

from minicnn.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
    ):
        super().__init__(params=list(params), lr=lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.velocities = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        updated = 0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            try:
                grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
                if self.grad_clip > 0.0:
                    norm = float(np.linalg.norm(grad))
                    if norm > self.grad_clip:
                        grad = grad * (self.grad_clip / norm)
                if self.momentum:
                    self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                    p.data = p.data + self.velocities[i]
                else:
                    p.data = p.data - self.lr * grad
                updated += 1
            except (ValueError, TypeError) as exc:
                import warnings
                warnings.warn(
                    f"SGD.step(): skipped param {i} ({getattr(p, 'name', None)!r}): {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
        return {'updated': updated, 'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay}
