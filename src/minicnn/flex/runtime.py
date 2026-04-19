from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_dir(cfg: dict[str, Any]) -> Path:
    project = cfg.get('project', {})
    root = Path(project.get('artifacts_root', 'artifacts'))
    run_name = project.get('run_name', 'default')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = root / f'{run_name}-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def dump_summary(run_dir: Path, summary: dict[str, Any]):
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
