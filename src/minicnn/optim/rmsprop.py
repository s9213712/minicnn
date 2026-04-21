from __future__ import annotations

import numpy as np

from minicnn.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        grad_clip: float = 0.0,
    ):
        super().__init__(params=list(params), lr=lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.grad_clip = grad_clip
        self.v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self.buf = [np.zeros_like(p.data, dtype=np.float32) for p in self.params] if momentum else None

    def step(self):
        updated = 0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            if self.grad_clip > 0.0:
                norm = float(np.linalg.norm(grad))
                if norm > self.grad_clip:
                    grad = grad * (self.grad_clip / norm)
            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * (grad * grad)
            step = grad / (np.sqrt(self.v[i]) + self.eps)
            if self.momentum and self.buf is not None:
                self.buf[i] = self.momentum * self.buf[i] + step
                p.data = p.data - self.lr * self.buf[i]
            else:
                p.data = p.data - self.lr * step
            updated += 1
        return {'updated': updated, 'lr': self.lr}
