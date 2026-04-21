"""Public API surface for cuda_native."""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
from minicnn.cuda_native.graph import NativeGraph, build_graph
from minicnn.cuda_native.validators import validate_cuda_native_model_config


def validate_cuda_native_config(cfg: dict[str, Any]) -> list[str]:
    """Validate a full experiment config for cuda_native compatibility.

    Returns a list of error strings (empty list = valid).
    """
    model_cfg = cfg.get('model', {})
    return validate_cuda_native_model_config(model_cfg)


def build_cuda_native_graph(
    model_cfg: dict[str, Any],
    input_shape: tuple[int, ...],
) -> NativeGraph:
    """Build and return a NativeGraph from a model config dict.

    Args:
        model_cfg:   dict with a 'layers' key (same format as flex/autograd).
        input_shape: fixed input shape, e.g. (1, 3, 32, 32).

    Raises:
        ValueError: if the config references unsupported ops or has bad attrs/shapes.
    """
    layers = model_cfg.get('layers', [])
    return build_graph(layers, input_shape)


def get_capability_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_cuda_native_capabilities()
