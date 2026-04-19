from __future__ import annotations

import numpy as np

from minicnn.nn.modules import Module, Sequential
from minicnn.nn.tensor import Parameter, Tensor
from minicnn.ops.nn_ops import batchnorm2d, conv2d, flatten, linear, maxpool2d, relu


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = np.sqrt(2.0 / max(in_features, 1))
        self.weight = self.add_parameter('weight', Parameter(np.random.randn(in_features, out_features).astype(np.float32) * scale))
        self.bias = self.add_parameter('bias', Parameter(np.zeros(out_features, dtype=np.float32))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return flatten(x)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        scale = np.sqrt(2.0 / max(in_channels * kernel_size * kernel_size, 1))
        self.weight = self.add_parameter(
            'weight',
            Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale),
        )
        self.bias = self.add_parameter('bias', Parameter(np.zeros(out_channels, dtype=np.float32))) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return maxpool2d(x, self.kernel_size, self.stride)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = self.add_parameter('weight', Parameter(np.ones(num_features, dtype=np.float32)))
        self.bias = self.add_parameter('bias', Parameter(np.zeros(num_features, dtype=np.float32)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return batchnorm2d(x, self.weight, self.bias, eps=self.eps)


class ResidualBlock(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = self.add_module('main', Sequential(
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(channels),
            ReLU(),
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(channels),
        ))
        self.activation = self.add_module('activation', ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.main(x) + x)
