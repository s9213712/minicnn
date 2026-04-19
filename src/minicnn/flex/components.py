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
