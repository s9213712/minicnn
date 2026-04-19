from __future__ import annotations

from pathlib import Path

from minicnn.config.schema import ExperimentConfig
from minicnn.training.factory import create_trainer


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.impl = create_trainer(config)

    def fit(self) -> Path:
        return self.impl.fit()
