from __future__ import annotations

from minicnn.engine.backends import CudaLegacyBackend, TorchLegacyBackend
from minicnn.framework.registry import GLOBAL_REGISTRY
from minicnn.optim.sgd import SGD
from minicnn.schedulers.plateau import ReduceLROnPlateau


def register_builtin_components():
    if GLOBAL_REGISTRY.summary():
        return
    GLOBAL_REGISTRY.register('backend', 'cuda', CudaLegacyBackend, 'Legacy handwritten CUDA adapter backend')
    GLOBAL_REGISTRY.register('backend', 'torch', TorchLegacyBackend, 'PyTorch baseline adapter backend')
    GLOBAL_REGISTRY.register('optimizer', 'sgd', SGD, 'Simple framework-layer SGD optimizer')
    GLOBAL_REGISTRY.register('scheduler', 'plateau', ReduceLROnPlateau, 'Reduce LR on plateau scheduler')
