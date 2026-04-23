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

            self.depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=bias,
            )
            self.norm = nn.LayerNorm(channels, eps=layer_norm_eps)
            self.pointwise1 = nn.Linear(channels, hidden_channels, bias=bias)
            self.activation = nn.GELU()
            self.pointwise2 = nn.Linear(hidden_channels, channels, bias=bias)
            if layer_scale_init_value > 0:
                self.layer_scale = nn.Parameter(torch.full((channels,), float(layer_scale_init_value)))
            else:
                self.layer_scale = None

        def forward(self, x):
            residual = x
            x = self.depthwise(x)
            # LayerNorm/MLP operate over channels-last features in ConvNeXt-style blocks.
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.pointwise1(x)
            x = self.activation(x)
            x = self.pointwise2(x)
            if self.layer_scale is not None:
                x = x * self.layer_scale
            x = x.permute(0, 3, 1, 2).contiguous()
            return residual + x

    # Layers
    register('layers', 'Conv2d')(nn.Conv2d)
    register('layers', 'Linear')(nn.Linear)
    register('layers', 'MaxPool2d')(nn.MaxPool2d)
    register('layers', 'AvgPool2d')(nn.AvgPool2d)
    register('layers', 'AdaptiveAvgPool2d')(nn.AdaptiveAvgPool2d)
    register('layers', 'BatchNorm2d')(nn.BatchNorm2d)
    register('layers', 'Dropout')(nn.Dropout)
    register('layers', 'Flatten')(nn.Flatten)
    register('layers', 'Identity')(nn.Identity)
    register('layers', 'GlobalAvgPool2d')(GlobalAvgPool2d)
    register('layers', 'ResidualBlock')(ResidualBlock)
    register('layers', 'ConvNeXtBlock')(ConvNeXtBlock)

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
