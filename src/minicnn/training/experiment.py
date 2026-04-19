from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from minicnn.config.schema import ExperimentConfig
from minicnn.paths import PROJECT_ROOT
from minicnn.utils import write_json, write_yaml


class ExperimentManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = PROJECT_ROOT / config.project.artifacts_root / f"{timestamp}_{config.project.run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self) -> None:
        write_yaml(self.run_dir / "config.yaml", asdict(self.config))
        write_json(self.run_dir / 'runtime_meta.json', {
            'project_name': self.config.project.name,
            'backend': self.config.backend.type,
            'legacy_entrypoint': self.config.backend.legacy_entrypoint,
        })
