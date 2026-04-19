from __future__ import annotations

import numpy as np


class Backend:
    name = 'base'

    def forward_linear(self, x, weight, bias=None):
        raise NotImplementedError

    def relu(self, x):
        raise NotImplementedError


class NumPyBackend(Backend):
    name = 'numpy'

    def forward_linear(self, x, weight, bias=None):
        out = x @ weight
        return out + bias if bias is not None else out

    def relu(self, x):
        return np.maximum(x, 0.0)


class TorchBackend(Backend):
    name = 'torch'

    def __init__(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('TorchBackend requires torch') from exc
        self.torch = torch

    def forward_linear(self, x, weight, bias=None):
        out = x @ weight
        return out + bias if bias is not None else out

    def relu(self, x):
        return self.torch.relu(x)


class CudaBackend(Backend):
    name = 'cuda'

    def forward_linear(self, x, weight, bias=None):
        raise NotImplementedError('CudaBackend direct runtime lowering is not implemented yet; use cuda_legacy trainer')

    def relu(self, x):
        raise NotImplementedError('CudaBackend direct runtime lowering is not implemented yet; use cuda_legacy trainer')
