from __future__ import annotations

from minicnn.config.schema import ExperimentConfig
from minicnn.engine.backends import CudaLegacyBackend, TorchLegacyBackend
from minicnn.framework import GLOBAL_REGISTRY
from minicnn.nn import Module
from minicnn.nn.tensor import Tensor
from minicnn.training.base_trainer import BaseTrainer
from minicnn.training.callbacks import (
    CheckpointManifestCallback,
    ConsoleLogger,
    FrameworkManifestCallback,
    JsonlMetricsCallback,
    SummaryCallback,
)


class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.add_parameter('weight', Tensor(data=1.0, requires_grad=True, name='weight'))


def build_callbacks(config: ExperimentConfig):
    callbacks = []
    if config.logging.console:
        callbacks.append(ConsoleLogger())
    if config.logging.jsonl:
        callbacks.append(JsonlMetricsCallback())
    callbacks.append(FrameworkManifestCallback())
    if config.callbacks.summary:
        callbacks.append(SummaryCallback())
    if config.callbacks.checkpoint:
        callbacks.append(CheckpointManifestCallback())
    return callbacks


def build_framework_stack(config: ExperimentConfig):
    model = DummyModel()
    optimizer = GLOBAL_REGISTRY.create(
        'optimizer',
        'sgd',
        model.parameters(),
        lr=config.optim.lr_fc,
        momentum=config.optim.momentum,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = None
    if config.scheduler.enabled:
        scheduler = GLOBAL_REGISTRY.create(
            'scheduler',
            config.scheduler.type,
            optimizer,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )
    return {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}


class CudaTrainer(BaseTrainer):
    def __init__(self, config: ExperimentConfig):
        super().__init__(
            config=config,
            backend=CudaLegacyBackend(framework_stack=build_framework_stack(config)),
            callbacks=build_callbacks(config),
        )


class TorchTrainer(BaseTrainer):
    def __init__(self, config: ExperimentConfig):
        super().__init__(
            config=config,
            backend=TorchLegacyBackend(framework_stack=build_framework_stack(config)),
            callbacks=build_callbacks(config),
        )
