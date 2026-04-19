from __future__ import annotations

from minicnn.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params=list(params), lr=lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        # Lightweight placeholder optimizer for the framework layer.
        # The handwritten CUDA path still performs legacy stepping inside the backend adapter.
        updated = 0
        for p in self.params:
            if p.grad is None:
                continue
            try:
                p.data = p.data - self.lr * p.grad
                updated += 1
            except Exception:
                # Keep optimizer layer non-fatal for metadata-only parameters.
                continue
        return {'updated': updated, 'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay}
