from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


def create_run_dir(cfg: dict[str, Any]) -> Path:
    project = cfg.get('project', {})
    root = Path(project.get('artifacts_root', 'artifacts'))
    run_name = project.get('run_name', 'default')
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        run_dir = root / f'{run_name}-{stamp}'
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:  # pragma: no cover - extremely unlikely, but keep collision-free
            continue
    else:  # pragma: no cover
        run_dir = root / f'{run_name}-{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}-{uuid4().hex[:8]}'
        run_dir.mkdir(parents=False, exist_ok=False)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def dump_summary(run_dir: Path, summary: dict[str, Any]):
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
