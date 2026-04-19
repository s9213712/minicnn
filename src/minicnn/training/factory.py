from __future__ import annotations

from minicnn.config.schema import ExperimentConfig
from minicnn.framework import GLOBAL_REGISTRY
from minicnn.training.trainers import CudaTrainer, TorchTrainer


def create_trainer(config: ExperimentConfig):
    backend = config.backend.type
    # Validate backend through registry so CLI can expose a unified component model.
    GLOBAL_REGISTRY.get('backend', backend)
    if backend == 'cuda':
        return CudaTrainer(config)
    if backend == 'torch':
        return TorchTrainer(config)
    raise ValueError(f'Unsupported backend: {backend}')
