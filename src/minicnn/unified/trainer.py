from __future__ import annotations

import json
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.flex.trainer import train_from_config
from minicnn.paths import BEST_MODELS_ROOT
from minicnn.unified.cuda_legacy import compile_to_legacy_experiment, summarize_legacy_mapping


_MANAGED_CUDA_ENV: dict[str, tuple[bool, str | None, str]] = {}


def _set_managed_env(name: str, value: Any | None) -> None:
    if value is None:
        state = _MANAGED_CUDA_ENV.pop(name, None)
        if state is not None:
            old_present, old_value, managed_value = state
            if os.environ.get(name) == managed_value:
                if old_present and old_value is not None:
                    os.environ[name] = old_value
                else:
                    os.environ.pop(name, None)
        return
    text = str(value)
    old_present, old_value, _old_managed = _MANAGED_CUDA_ENV.get(
        name,
        (name in os.environ, os.environ.get(name), ''),
    )
    os.environ[name] = text
    _MANAGED_CUDA_ENV[name] = (old_present, old_value, text)


def _configure_cuda_legacy_runtime(cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    runtime = cfg.get('runtime', {})
    cuda_variant = runtime.get('cuda_variant')
    cuda_so = runtime.get('cuda_so')
    _set_managed_env('MINICNN_CUDA_VARIANT', cuda_variant)
    _set_managed_env('MINICNN_CUDA_SO', cuda_so)
    if cuda_variant is not None:
        summary['cuda_variant'] = str(cuda_variant)
    if cuda_so is not None:
        summary['cuda_so'] = str(cuda_so)


def _reload_legacy_modules_after_config() -> None:
    from minicnn.core.cuda_backend import reset_library_cache

    reset_library_cache()
    for name in (
        'minicnn.core.cuda_backend',
        'minicnn.training.evaluation',
        'minicnn.training.cuda_ops',
        'minicnn.training.cuda_workspace',
        'minicnn.training.cuda_batch',
        'minicnn.training.checkpoints',
        'minicnn.training.train_cuda',
    ):
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)


def train_unified_from_config(cfg: dict[str, Any]) -> Path:
    backend = str(cfg.get('engine', {}).get('backend', 'torch'))

    if backend == 'torch':
        run_dir = train_from_config(cfg)
        torch_summary_path = run_dir / 'summary.json'
        # Start from the existing flex summary so its fields are preserved.
        merged: dict[str, Any] = json.loads(torch_summary_path.read_text(encoding='utf-8')) if torch_summary_path.exists() else {}
        # Merge in unified metadata without overwriting existing keys.
        merged.setdefault('selected_backend', backend)
        merged.setdefault('effective_backend', 'torch')
        merged.setdefault('run_dir', str(run_dir))
        merged.setdefault('best_model_path', str(BEST_MODELS_ROOT / f'{run_dir.name}_best.pt'))
        merged['config_backend_toggle_only'] = True
        dump_summary(run_dir, merged)
        return run_dir

    if backend == 'cuda_legacy':
        run_dir = create_run_dir(cfg)
        cuda_summary: dict[str, Any] = {
            'selected_backend': backend,
            'run_dir': str(run_dir),
            'config_backend_toggle_only': True,
        }
        _configure_cuda_legacy_runtime(cfg, cuda_summary)
        exp = compile_to_legacy_experiment(cfg)
        os.environ['MINICNN_ARTIFACT_RUN_DIR'] = str(run_dir)
        from minicnn.config.settings import apply_experiment_config
        apply_experiment_config(exp)
        _reload_legacy_modules_after_config()
        from minicnn.training.train_cuda import main as legacy_main
        legacy_result = legacy_main() or {}
        cuda_summary['effective_backend'] = 'cuda_legacy'
        if isinstance(legacy_result, dict):
            for key in ('test_loss', 'test_acc'):
                if key in legacy_result:
                    cuda_summary[key] = legacy_result[key]
            best_model_path = legacy_result.get('best_model_path')
        else:
            best_model_path = None
        cuda_summary['best_model_path'] = str(best_model_path or '')
        cuda_summary['legacy_mapping'] = summarize_legacy_mapping(cfg)
        dump_summary(run_dir, cuda_summary)
        return run_dir

    if backend == 'cuda_native':
        from minicnn.unified.cuda_native import run_cuda_native_training
        import warnings
        warnings.warn(
            'engine.backend=cuda_native is experimental and forward-only. '
            'Training loop is a research prototype, not production-ready.',
            stacklevel=2,
        )
        return run_cuda_native_training(cfg)

    raise ValueError(f'Unknown engine.backend={backend!r}; expected torch, cuda_legacy, or cuda_native')
