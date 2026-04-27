from __future__ import annotations

import numpy as np

from minicnn.nn.tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError

    def _clip_gradients(self, grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
        if max_norm <= 0.0 or not grads:
            return grads
        total_norm_sq = sum(float(np.square(grad, dtype=np.float64).sum()) for grad in grads)
        if total_norm_sq <= 0.0:
            return grads
        total_norm = total_norm_sq ** 0.5
        if total_norm <= max_norm:
            return grads
        scale = max_norm / (total_norm + 1e-12)
        return [grad * scale for grad in grads]

    def _prepared_grads(
        self,
        *,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        decoupled_weight_decay: bool = False,
    ) -> list[tuple[int, Tensor, np.ndarray]]:
        entries: list[tuple[int, Tensor]] = []
        grads: list[np.ndarray] = []
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = np.asarray(p.grad, dtype=np.float32)
            if grad.shape != p.data.shape:
                raise ValueError(
                    f'Gradient shape {grad.shape} does not match parameter shape {p.data.shape} for param {i}'
                )
            if not decoupled_weight_decay and weight_decay:
                grad = grad + weight_decay * p.data
            entries.append((i, p))
            grads.append(grad)
        clipped = self._clip_gradients(grads, grad_clip)
        return [(i, p, grad) for (i, p), grad in zip(entries, clipped)]

    def state_dict(self) -> dict[str, object]:
        return {'lr': self.lr, 'num_params': len(self.params)}
