"""Bridge: shared config → cuda_native backend.

Translates the project's unified experiment config into cuda_native API calls.
Phase 0 stub — validation only, no graph construction yet.
"""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.api import validate_cuda_native_config, get_capability_summary


def check_config(cfg: dict[str, Any]) -> list[str]:
    """Return validation errors for *cfg* against cuda_native constraints."""
    return validate_cuda_native_config(cfg)


def get_summary() -> dict[str, object]:
    """Return the cuda_native capability summary."""
    return get_capability_summary()
