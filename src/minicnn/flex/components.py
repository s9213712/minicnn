from __future__ import annotations

from .registry import register

try:
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
except Exception:  # pragma: no cover
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

    # Activations
    register('activations', 'ReLU')(nn.ReLU)
    register('activations', 'LeakyReLU')(nn.LeakyReLU)
    register('activations', 'GELU')(nn.GELU)
    register('activations', 'SiLU')(nn.SiLU)
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
