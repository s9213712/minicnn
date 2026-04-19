from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minicnn.config.schema import ExperimentConfig


@dataclass
class TrainContext:
    config: ExperimentConfig
    run_dir: Path
    backend_name: str
    profiler_enabled: bool = False
    state: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
