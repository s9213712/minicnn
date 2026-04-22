from __future__ import annotations

from pathlib import Path

from minicnn.config import settings
from minicnn.core.cuda_backend import check_cuda_ready, resolve_library_path
from minicnn.data.cifar10 import cifar10_ready
from minicnn.flex.registry import describe_registries
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT
from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED

DIAGNOSTIC_SCHEMA_VERSION = 1


def _check(name: str, ok: bool, *, required: bool = True, details: dict[str, object] | None = None, suggested_fix: str | None = None) -> dict[str, object]:
    return {
        'name': name,
        'ok': bool(ok),
        'required': required,
        'severity': 'info' if ok else ('error' if required else 'warning'),
        'details': details or {},
        'suggested_fix': suggested_fix or '',
    }


def _summary_status(checks: list[dict[str, object]]) -> str:
    if any((not bool(check['ok'])) and bool(check['required']) for check in checks):
        return 'error'
    if any(not bool(check['ok']) for check in checks):
        return 'warning'
    return 'ok'


def _warning_messages(checks: list[dict[str, object]]) -> list[str]:
    return [
        str(check['name'])
        for check in checks
        if (not bool(check['ok'])) and (not bool(check['required']))
    ]


def _error_messages(checks: list[dict[str, object]]) -> list[str]:
    return [
        str(check['name'])
        for check in checks
        if (not bool(check['ok'])) and bool(check['required'])
    ]


def _check_summary(checks: list[dict[str, object]]) -> dict[str, int]:
    return {
        'total': len(checks),
        'ok': sum(1 for check in checks if bool(check['ok'])),
        'warning': sum(1 for check in checks if (not bool(check['ok'])) and (not bool(check['required']))),
        'error': sum(1 for check in checks if (not bool(check['ok'])) and bool(check['required'])),
    }


def build_diagnostic_payload(*, checks: list[dict[str, object]], extra: dict[str, object] | None = None) -> dict[str, object]:
    status = _summary_status(checks)
    return {
        'schema_version': DIAGNOSTIC_SCHEMA_VERSION,
        'status': status,
        'summary_status': status,
        'check_summary': _check_summary(checks),
        'checks': checks,
        'warnings': _warning_messages(checks),
        'errors': _error_messages(checks),
        **(extra or {}),
    }


def healthcheck() -> dict[str, object]:
    shared_candidates = sorted([p.name for p in CPP_ROOT.glob('*.so')])
    checks = [
        _check(
            'project_root',
            PROJECT_ROOT.exists(),
            details={'project_root': str(PROJECT_ROOT)},
        ),
        _check(
            'cpp_root',
            CPP_ROOT.exists(),
            details={'cpp_root': str(CPP_ROOT)},
        ),
        _check(
            'native_cuda_artifacts',
            bool(shared_candidates),
            required=False,
            details={'shared_objects': shared_candidates},
            suggested_fix='Run minicnn build --legacy-make --check if you need cuda_legacy.',
        ),
        _check(
            'cifar10_data',
            DATA_ROOT.exists(),
            required=False,
            details={'data_root': str(DATA_ROOT)},
            suggested_fix='Run minicnn prepare-data if you need the handcrafted CIFAR-10 path.',
        ),
    ]
    return build_diagnostic_payload(checks=checks, extra={
        'project_root_exists': PROJECT_ROOT.exists(),
        'data_root_exists': DATA_ROOT.exists(),
        'cpp_root_exists': CPP_ROOT.exists(),
        'shared_objects': shared_candidates,
        'flex_registries': describe_registries(),
        'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED,
    })


def doctor() -> dict[str, object]:
    native_path = resolve_library_path()
    cuda = check_cuda_ready(native_path)
    from minicnn.cuda_native.api import get_capability_summary
    cuda_native_caps = get_capability_summary()
    checks = [
        _check(
            'native_cuda_library',
            Path(native_path).exists(),
            required=False,
            details={'path': native_path},
            suggested_fix='Run minicnn build --legacy-make --check if you want the handcrafted CUDA backend.',
        ),
        _check(
            'cifar10_data',
            cifar10_ready(DATA_ROOT),
            required=False,
            details={'data_root': str(DATA_ROOT)},
            suggested_fix='Run minicnn prepare-data to enable the handcrafted CIFAR-10 backend.',
        ),
    ]
    return build_diagnostic_payload(checks=checks, extra={
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
    })
