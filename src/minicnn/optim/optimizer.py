from __future__ import annotations

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

    def state_dict(self) -> dict[str, object]:
        return {'lr': self.lr, 'num_params': len(self.params)}
