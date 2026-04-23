from __future__ import annotations

from minicnn.nn import AvgPool2d, BatchNorm2d, Conv2d, Dropout, Flatten, LeakyReLU, Linear, MaxPool2d, ReLU, ResidualBlock, Sequential, Sigmoid, SiLU, Tanh

MODEL_REGISTRY = {
    'AvgPool2d': AvgPool2d,
    'BatchNorm2d': BatchNorm2d,
    'Conv2d': Conv2d,
    'Dropout': Dropout,
    'Flatten': Flatten,
    'LeakyReLU': LeakyReLU,
    'Linear': Linear,
    'MaxPool2d': MaxPool2d,
    'ReLU': ReLU,
    'ResidualBlock': ResidualBlock,
    'Sequential': Sequential,
    'Sigmoid': Sigmoid,
    'SiLU': SiLU,
    'Tanh': Tanh,
}


def list_model_components() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_model_component(name: str):
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ', '.join(list_model_components())
        raise KeyError(f'Unknown MiniCNN model component {name!r}; expected one of: {choices}') from exc
