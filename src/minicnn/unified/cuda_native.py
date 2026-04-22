"""Bridge: shared unified config → cuda_native backend.

Responsibilities:
- Validate the config against cuda_native constraints (clear errors, no silent fallback)
- Translate config layers into a NativeGraph
- Initialize parameters
- Load dataset as numpy arrays (no torch dependency)
- Run minimal training loop and return a summary dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from minicnn.cuda_native.api import (
    build_cuda_native_graph,
    get_capability_summary,
    validate_cuda_native_config,
)
from minicnn.unified._cuda_native_runtime import train_and_summarize_native_backend


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def check_config(cfg: dict[str, Any]) -> list[str]:
    """Return validation errors for *cfg* against cuda_native constraints."""
    return validate_cuda_native_config(cfg)


def get_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_capability_summary()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_cuda_native_training(cfg: dict[str, Any]) -> Path:
    """Run a full training loop for cuda_native and return the run directory.

    This is the backend-specific training entry point wired into
    unified/trainer.py when engine.backend = cuda_native.
    """
    errors = check_config(cfg)
    if errors:
        raise ValueError(
            'Config is not compatible with cuda_native:\n- ' + '\n- '.join(errors)
        )

    model_cfg = cfg.get('model', {})
    batch_size = int(cfg.get('train', {}).get('batch_size', 64))
    input_shape = tuple(cfg.get('dataset', {}).get('input_shape', [3, 32, 32]))
    graph = build_cuda_native_graph(model_cfg, (batch_size, *input_shape))
    return train_and_summarize_native_backend(
        cfg,
        graph=graph,
        capabilities=get_capability_summary(),
    )
