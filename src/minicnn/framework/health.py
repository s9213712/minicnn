from __future__ import annotations

from pathlib import Path

from minicnn.config import settings
from minicnn.core.cuda_backend import check_cuda_ready, resolve_library_path
from minicnn.data.cifar10 import cifar10_ready
from minicnn.flex.registry import describe_registries
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT
from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED


def healthcheck() -> dict[str, object]:
    shared_candidates = sorted([p.name for p in CPP_ROOT.glob('*.so')])
    return {
        'project_root_exists': PROJECT_ROOT.exists(),
        'data_root_exists': DATA_ROOT.exists(),
        'cpp_root_exists': CPP_ROOT.exists(),
        'shared_objects': shared_candidates,
        'flex_registries': describe_registries(),
        'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED,
    }


def doctor() -> dict[str, object]:
    native_path = resolve_library_path()
    cuda = check_cuda_ready(native_path)
    from minicnn.cuda_native.api import get_capability_summary
    cuda_native_caps = get_capability_summary()
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
        'cuda_native': cuda_native_caps,
        'data': {
            'cifar10_ready': cifar10_ready(DATA_ROOT),
        },
        'settings': settings.summarize(),
        'flex_registries': describe_registries(),
        'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED,
    }
