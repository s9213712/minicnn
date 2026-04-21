from __future__ import annotations

import numpy as np

from minicnn.optim.optimizer import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip: float = 0.0,
    ):
        super().__init__(params=list(params), lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.t = 0
        self.m = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        self.t += 1
        updated = 0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.copy()
            if self.grad_clip > 0.0:
                norm = float(np.linalg.norm(grad))
                if norm > self.grad_clip:
                    grad = grad * (self.grad_clip / norm)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay:
                p.data = p.data - self.lr * self.weight_decay * p.data
            updated += 1
        return {'updated': updated, 'lr': self.lr, 't': self.t}
