from __future__ import annotations

from pathlib import Path

from minicnn.framework.registry import GLOBAL_REGISTRY
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT


def healthcheck() -> dict[str, object]:
    shared_candidates = sorted([p.name for p in CPP_ROOT.glob('*.so')])
    return {
        'project_root_exists': PROJECT_ROOT.exists(),
        'data_root_exists': DATA_ROOT.exists(),
        'cpp_root_exists': CPP_ROOT.exists(),
        'shared_objects': shared_candidates,
        'registered_components': GLOBAL_REGISTRY.summary(),
    }
