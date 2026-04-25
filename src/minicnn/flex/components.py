from __future__ import annotations

from .registry import register

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    lr_scheduler = None


if nn is not None:
    class DepthwiseConv2d(nn.Conv2d):
        def __init__(
            self,
            in_channels: int,
            kernel_size: int | tuple[int, int] = 3,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] | None = None,
            dilation: int | tuple[int, int] = 1,
            out_channels: int | None = None,
            channel_multiplier: int = 1,
            bias: bool = True,
        ):
            if int(in_channels) <= 0:
                raise ValueError('DepthwiseConv2d in_channels must be positive')
            if out_channels is None:
                out_channels = int(in_channels) * int(channel_multiplier)
            if int(out_channels) <= 0:
                raise ValueError('DepthwiseConv2d out_channels must be positive')
            if int(out_channels) % int(in_channels) != 0:
                raise ValueError('DepthwiseConv2d out_channels must be a multiple of in_channels')
            if padding is None:
                if isinstance(kernel_size, tuple):
                    padding = tuple(int(k) // 2 for k in kernel_size)
                else:
                    padding = int(kernel_size) // 2
            super().__init__(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=int(in_channels),
                bias=bias,
            )


    class PointwiseConv2d(nn.Conv2d):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
        ):
            if int(out_channels) <= 0:
                raise ValueError('PointwiseConv2d out_channels must be positive')
            super().__init__(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )


    class LayerNorm2d(nn.Module):
        def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
        ):
            super().__init__()
            num_channels = int(num_channels)
            if num_channels <= 0:
                raise ValueError('LayerNorm2d requires a positive num_channels value')
            self.num_channels = num_channels
            self.eps = float(eps)
            if affine:
                self.weight = nn.Parameter(torch.ones(num_channels))
                self.bias = nn.Parameter(torch.zeros(num_channels))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

        def forward(self, x):
            if x.ndim != 4:
                raise ValueError(f'LayerNorm2d expects NCHW input, got shape {tuple(x.shape)}')
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            x = (x - mean) * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                x = x * self.weight.view(1, -1, 1, 1)
            if self.bias is not None:
                x = x + self.bias.view(1, -1, 1, 1)
            return x


    class GlobalAvgPool2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            return self.pool(x)


    class ResidualBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int | None = None,
            channels: int | None = None,
            stride: int = 1,
            kernel_size: int = 3,
            padding: int | None = None,
            bias: bool = False,
            activation: str = 'ReLU',
        ):
            super().__init__()
            out_channels = int(out_channels if out_channels is not None else channels if channels is not None else in_channels)
            padding = kernel_size // 2 if padding is None else padding
            act_cls = getattr(nn, activation)
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                act_cls(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()
            self.activation = act_cls()

        def forward(self, x):
            return self.activation(self.main(x) + self.shortcut(x))


    class ConvNeXtBlock(nn.Module):
        def __init__(
            self,
            channels: int | None = None,
            in_channels: int | None = None,
            kernel_size: int = 7,
            expansion_ratio: float = 4.0,
            hidden_channels: int | None = None,
            layer_scale_init_value: float = 1e-6,
            layer_norm_eps: float = 1e-6,
            bias: bool = True,
        ):
            super().__init__()
            channels = int(channels if channels is not None else in_channels if in_channels is not None else 0)
            if channels <= 0:
                raise ValueError('ConvNeXtBlock requires a positive channels value')
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError('ConvNeXtBlock kernel_size must be a positive odd integer')
            if hidden_channels is None:
                hidden_channels = int(round(channels * float(expansion_ratio)))
            hidden_channels = int(hidden_channels)
            if hidden_channels <= 0:
                raise ValueError('ConvNeXtBlock hidden_channels must be positive')

            self.depthwise = DepthwiseConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                bias=bias,
            )
            self.norm = LayerNorm2d(channels, eps=layer_norm_eps)
            self.pointwise1 = PointwiseConv2d(channels, hidden_channels, bias=bias)
            self.activation = nn.GELU()
            self.pointwise2 = PointwiseConv2d(hidden_channels, channels, bias=bias)
            if layer_scale_init_value > 0:
                self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init_value)))
            else:
                self.layer_scale = None

        def forward(self, x):
            residual = x
            x = self.depthwise(x)
            x = self.norm(x)
            x = self.pointwise1(x)
            x = self.activation(x)
            x = self.pointwise2(x)
            if self.layer_scale is not None:
                x = x * self.layer_scale
            return residual + x

    # Layers
    register('layers', 'Conv2d')(nn.Conv2d)
    register('layers', 'DepthwiseConv2d')(DepthwiseConv2d)
    register('layers', 'depthwise_conv2d')(DepthwiseConv2d)
    register('layers', 'PointwiseConv2d')(PointwiseConv2d)
    register('layers', 'pointwise_conv2d')(PointwiseConv2d)
    register('layers', 'Linear')(nn.Linear)
    register('layers', 'MaxPool2d')(nn.MaxPool2d)
    register('layers', 'AvgPool2d')(nn.AvgPool2d)
    register('layers', 'AdaptiveAvgPool2d')(nn.AdaptiveAvgPool2d)
    register('layers', 'BatchNorm2d')(nn.BatchNorm2d)
    register('layers', 'LayerNorm2d')(LayerNorm2d)
    register('layers', 'layernorm2d')(LayerNorm2d)
    register('layers', 'Dropout')(nn.Dropout)
    register('layers', 'Flatten')(nn.Flatten)
    register('layers', 'Identity')(nn.Identity)
    register('layers', 'GlobalAvgPool2d')(GlobalAvgPool2d)
    register('layers', 'ResidualBlock')(ResidualBlock)
    register('layers', 'ConvNeXtBlock')(ConvNeXtBlock)
    register('layers', 'convnext_block')(ConvNeXtBlock)

    # Activations
    register('activations', 'ReLU')(nn.ReLU)
    register('activations', 'LeakyReLU')(nn.LeakyReLU)
    register('activations', 'GELU')(nn.GELU)
    register('activations', 'SiLU')(nn.SiLU)
    register('activations', 'Sigmoid')(nn.Sigmoid)
    register('activations', 'Tanh')(nn.Tanh)

    # Losses
    register('losses', 'CrossEntropyLoss')(nn.CrossEntropyLoss)
    register('losses', 'MSELoss')(nn.MSELoss)
    register('losses', 'BCEWithLogitsLoss')(nn.BCEWithLogitsLoss)

if optim is not None:
    # Optimizers
    register('optimizers', 'SGD')(optim.SGD)
    register('optimizers', 'Adam')(optim.Adam)
    register('optimizers', 'AdamW')(optim.AdamW)

if lr_scheduler is not None:
    # Schedulers
    register('schedulers', 'StepLR')(lr_scheduler.StepLR)
    register('schedulers', 'CosineAnnealingLR')(lr_scheduler.CosineAnnealingLR)
    register('schedulers', 'ReduceLROnPlateau')(lr_scheduler.ReduceLROnPlateau)
