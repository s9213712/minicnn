from __future__ import annotations

from pathlib import Path

from minicnn.config import settings
from minicnn.core.cuda_backend import check_cuda_ready, resolve_library_path
from minicnn.data.cifar10 import cifar10_ready
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


def doctor() -> dict[str, object]:
    native_path = resolve_library_path()
    cuda = check_cuda_ready(native_path)
    return {
        'project': {
            'project_root': str(PROJECT_ROOT),
            'project_root_exists': PROJECT_ROOT.exists(),
            'cpp_root': str(CPP_ROOT),
            'cpp_root_exists': CPP_ROOT.exists(),
            'data_root': str(DATA_ROOT),
            'data_root_exists': DATA_ROOT.exists(),
        },
        'native_cuda': {
            'path': native_path,
            'artifact_present': Path(native_path).exists(),
            **cuda,
        },
        'data': {
            'cifar10_ready': cifar10_ready(DATA_ROOT),
        },
        'settings': settings.summarize(),
        'registries': GLOBAL_REGISTRY.summary(),
    }
