"""Public API surface for cuda_native.

Phase 0: stubs that establish the interface contract.
Actual graph construction is wired in Phase 1.
"""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
from minicnn.cuda_native.validators import validate_cuda_native_model_config


def validate_cuda_native_config(cfg: dict[str, Any]) -> list[str]:
    """Validate a full experiment config for cuda_native compatibility.

    Returns a list of error strings (empty list = valid).
    """
    model_cfg = cfg.get('model', {})
    return validate_cuda_native_model_config(model_cfg)


def build_cuda_native_graph(model_cfg: dict[str, Any], input_shape: tuple[int, ...]):
    """Build a NativeGraph from a model config dict.

    Phase 0 stub — raises NotImplementedError until Phase 1 is wired.
    """
    errors = validate_cuda_native_model_config(model_cfg)
    if errors:
        raise ValueError('cuda_native validation failed:\n- ' + '\n- '.join(errors))
    raise NotImplementedError(
        'cuda_native graph construction is not yet implemented (Phase 1). '
        'Capabilities: ' + str(get_cuda_native_capabilities())
    )


def get_capability_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_cuda_native_capabilities()
