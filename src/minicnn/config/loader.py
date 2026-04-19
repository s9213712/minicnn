from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    BackendConfig,
    CallbackConfig,
    ExperimentConfig,
    FrameworkConfig,
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


def _parse_scalar(text: str) -> Any:
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if text.startswith('0') and text != '0' and not text.startswith('0.'):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith('[') and text.endswith(']'):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(',')]
    return text


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    data: dict[str, Any] = ExperimentConfig().to_dict()
    if path:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError("Config file must contain a mapping at the top level")
        _deep_update(data, loaded)
    if overrides:
        for item in overrides:
            if '=' not in item:
                raise ValueError(f"Override must look like section.key=value, got: {item}")
            key, raw = item.split('=', 1)
            parts = key.split('.')
            cur = data
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = _parse_scalar(raw)
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
