from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
