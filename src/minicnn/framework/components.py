from __future__ import annotations

from minicnn.engine.backends import CudaLegacyBackend, TorchLegacyBackend
from minicnn.framework.registry import GLOBAL_REGISTRY
from minicnn.optim.adam import Adam
from minicnn.optim.adamw import AdamW
from minicnn.optim.rmsprop import RMSprop
from minicnn.optim.sgd import SGD
from minicnn.nn.layers import LeakyReLU, ReLU, Sigmoid, SiLU, Tanh
from minicnn.schedulers.cosine import CosineAnnealingLR
from minicnn.schedulers.plateau import ReduceLROnPlateau
from minicnn.schedulers.step import StepLR


def register_builtin_components():
    if GLOBAL_REGISTRY.summary():
        return
    GLOBAL_REGISTRY.register('backend', 'cuda', CudaLegacyBackend, 'Legacy handwritten CUDA adapter backend')
    GLOBAL_REGISTRY.register('backend', 'torch', TorchLegacyBackend, 'PyTorch baseline adapter backend')
    GLOBAL_REGISTRY.register('optimizer', 'sgd', SGD, 'SGD optimizer')
    GLOBAL_REGISTRY.register('optimizer', 'adam', Adam, 'Adam optimizer')
    GLOBAL_REGISTRY.register('optimizer', 'adamw', AdamW, 'AdamW optimizer with decoupled weight decay')
    GLOBAL_REGISTRY.register('optimizer', 'rmsprop', RMSprop, 'RMSprop optimizer')
    GLOBAL_REGISTRY.register('scheduler', 'plateau', ReduceLROnPlateau, 'Reduce LR on plateau scheduler')
    GLOBAL_REGISTRY.register('scheduler', 'step', StepLR, 'Step LR scheduler')
    GLOBAL_REGISTRY.register('scheduler', 'cosine', CosineAnnealingLR, 'Cosine annealing LR scheduler')
    GLOBAL_REGISTRY.register('activation', 'relu', ReLU, 'ReLU activation')
    GLOBAL_REGISTRY.register('activation', 'leaky_relu', LeakyReLU, 'LeakyReLU activation')
    GLOBAL_REGISTRY.register('activation', 'silu', SiLU, 'SiLU activation')
    GLOBAL_REGISTRY.register('activation', 'sigmoid', Sigmoid, 'Sigmoid activation')
    GLOBAL_REGISTRY.register('activation', 'tanh', Tanh, 'Tanh activation')


def get_supported_components() -> dict[str, list[str]]:
    register_builtin_components()
    return GLOBAL_REGISTRY.summary()
