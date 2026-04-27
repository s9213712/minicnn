from __future__ import annotations

import numpy as np

from minicnn.nn.modules import Module, Sequential
from minicnn.nn.tensor import Parameter, Tensor
from minicnn.ops.nn_ops import avgpool2d, batchnorm2d, conv2d, dropout, flatten, leaky_relu, linear, maxpool2d, relu, sigmoid, silu, tanh
from minicnn.random import get_global_rng


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or get_global_rng()
        scale = np.sqrt(2.0 / max(in_features, 1))
        self.weight = self.add_parameter('weight', Parameter(rng.standard_normal((in_features, out_features)).astype(np.float32) * scale))
        self.bias = self.add_parameter('bias', Parameter(np.zeros(out_features, dtype=np.float32))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.negative_slope)


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return dropout(x, p=self.p, training=self.training)


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return flatten(x)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        rng = rng or get_global_rng()
        scale = np.sqrt(2.0 / max(in_channels * kernel_size * kernel_size, 1))
        self.weight = self.add_parameter(
            'weight',
            Parameter(rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32) * scale),
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


class AvgPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return avgpool2d(x, self.kernel_size, self.stride)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.weight = self.add_parameter('weight', Parameter(np.ones(num_features, dtype=np.float32)))
        self.bias = self.add_parameter('bias', Parameter(np.zeros(num_features, dtype=np.float32)))
        self.running_mean = self.register_buffer('running_mean', np.zeros(num_features, dtype=np.float32))
        self.running_var = self.register_buffer('running_var', np.ones(num_features, dtype=np.float32))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        return batchnorm2d(
            x,
            self.weight,
            self.bias,
            eps=self.eps,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=self.training,
            momentum=self.momentum,
        )


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
