from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.flex.trainer import train_from_config
from minicnn.unified.cuda_legacy import compile_to_legacy_experiment, summarize_legacy_mapping


def train_unified_from_config(cfg: dict[str, Any]) -> Path:
    backend = str(cfg.get('engine', {}).get('backend', 'torch'))

    if backend == 'torch':
        run_dir = train_from_config(cfg)
        summary: dict[str, Any] = {
            'selected_backend': backend,
            'effective_backend': 'torch',
            'run_dir': str(run_dir),
            'config_backend_toggle_only': True,
        }
        dump_summary(run_dir, summary)
        return run_dir

    if backend == 'cuda_legacy':
        run_dir = create_run_dir(cfg)
        summary: dict[str, Any] = {
            'selected_backend': backend,
            'run_dir': str(run_dir),
            'config_backend_toggle_only': True,
        }
        exp = compile_to_legacy_experiment(cfg)
        os.environ['MINICNN_ARTIFACT_RUN_DIR'] = str(run_dir)
        from minicnn.config.settings import apply_experiment_config
        apply_experiment_config(exp)
        from minicnn.training.train_cuda import main as legacy_main
        legacy_main()
        summary['effective_backend'] = 'cuda_legacy'
        summary['legacy_mapping'] = summarize_legacy_mapping(cfg)
        dump_summary(run_dir, summary)
        return run_dir

    raise ValueError(f'Unknown engine.backend={backend!r}; expected torch or cuda_legacy')
