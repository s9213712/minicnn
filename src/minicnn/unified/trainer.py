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


def _configure_cuda_legacy_runtime(cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    runtime = cfg.get('runtime', {})
    cuda_variant = runtime.get('cuda_variant')
    cuda_so = runtime.get('cuda_so')
    if cuda_variant is not None:
        os.environ['MINICNN_CUDA_VARIANT'] = str(cuda_variant)
        summary['cuda_variant'] = str(cuda_variant)
    if cuda_so is not None:
        os.environ['MINICNN_CUDA_SO'] = str(cuda_so)
        summary['cuda_so'] = str(cuda_so)


def _reload_legacy_modules_after_config() -> None:
    for name in (
        'minicnn.training.evaluation',
        'minicnn.training.cuda_ops',
        'minicnn.training.cuda_workspace',
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
        torch_summary = json.loads(torch_summary_path.read_text(encoding='utf-8')) if torch_summary_path.exists() else {}
        summary: dict[str, Any] = {
            'selected_backend': backend,
            'effective_backend': 'torch',
            'run_dir': str(run_dir),
            'best_model_path': str(BEST_MODELS_ROOT / f'{run_dir.name}_best.pt'),
            'config_backend_toggle_only': True,
        }
        for key in ('test_loss', 'test_acc'):
            if key in torch_summary:
                summary[key] = torch_summary[key]
        dump_summary(run_dir, summary)
        return run_dir

    if backend == 'cuda_legacy':
        run_dir = create_run_dir(cfg)
        summary: dict[str, Any] = {
            'selected_backend': backend,
            'run_dir': str(run_dir),
            'config_backend_toggle_only': True,
        }
        _configure_cuda_legacy_runtime(cfg, summary)
        exp = compile_to_legacy_experiment(cfg)
        os.environ['MINICNN_ARTIFACT_RUN_DIR'] = str(run_dir)
        from minicnn.config.settings import apply_experiment_config
        apply_experiment_config(exp)
        _reload_legacy_modules_after_config()
        from minicnn.training.train_cuda import main as legacy_main
        legacy_main()
        summary['effective_backend'] = 'cuda_legacy'
        summary['best_model_path'] = str(legacy_main.__globals__.get('BEST_MODEL_PATH', ''))
        summary['legacy_mapping'] = summarize_legacy_mapping(cfg)
        dump_summary(run_dir, summary)
        return run_dir

    raise ValueError(f'Unknown engine.backend={backend!r}; expected torch or cuda_legacy')
