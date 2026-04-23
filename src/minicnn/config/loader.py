from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from minicnn.config.parsing import parse_override_parts, parse_scalar, set_nested_value
from .schema import (
    BackendConfig,
    CallbackConfig,
    ExperimentConfig,
    FrameworkConfig,
    LossConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    ProjectConfig,
    RuntimeConfig,
    SchedulerConfig,
    TrainConfig,
)


SECTION_TYPES = {
    "project": ProjectConfig,
    "backend": BackendConfig,
    "train": TrainConfig,
    "optim": OptimConfig,
    "loss": LossConfig,
    "model": ModelConfig,
    "runtime": RuntimeConfig,
    "logging": LoggingConfig,
    "callbacks": CallbackConfig,
    "framework": FrameworkConfig,
    "scheduler": SchedulerConfig,
}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
    parsed_overrides: list[tuple[list[str], Any]] = []
    for item in overrides:
        parts, raw = parse_override_parts(item)
        parsed_overrides.append((parts, parse_scalar(raw)))

    parsed_overrides.sort(key=lambda item: 0 if item[0][-1] == 'type' else 1)
    for parts, value in parsed_overrides:
        set_nested_value(data, parts, value, clear_on_type_change=True)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data: dict[str, Any] = ExperimentConfig().to_dict()
    if path:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError("Config file must contain a mapping at the top level")
        _deep_update(data, loaded)
    if overrides:
        _apply_overrides(data, overrides)
    return dict_to_config(data)


def dict_to_config(data: dict[str, Any]) -> ExperimentConfig:
    kwargs = {}
    for section, cls in SECTION_TYPES.items():
        section_data = copy.deepcopy(data.get(section, {}))
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(section_data) - valid)
        if unknown:
            raise ValueError(f"Unknown keys in section '{section}': {unknown}")
        kwargs[section] = cls(**section_data)
    return ExperimentConfig(**kwargs)
