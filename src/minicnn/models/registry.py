from __future__ import annotations

from minicnn.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, ReLU, ResidualBlock, Sequential

MODEL_REGISTRY = {
    'BatchNorm2d': BatchNorm2d,
    'Conv2d': Conv2d,
    'Flatten': Flatten,
    'Linear': Linear,
    'MaxPool2d': MaxPool2d,
    'ReLU': ReLU,
    'ResidualBlock': ResidualBlock,
    'Sequential': Sequential,
}


def get_model_component(name: str):
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ', '.join(sorted(MODEL_REGISTRY))
        raise KeyError(f'Unknown MiniCNN model component {name!r}; expected one of: {choices}') from exc
