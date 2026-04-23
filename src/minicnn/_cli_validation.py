from __future__ import annotations

from minicnn._cli_config import _load_unified_config_or_exit
from minicnn._cli_output import _print_json, _print_validation_result
from minicnn.backend_capability import validate_backend_model_capabilities


def handle_validate_config(args) -> int:
    from minicnn.cuda_native.api import validate_cuda_native_config
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility

    cfg = _load_unified_config_or_exit(args.config, args.overrides)
    backend = cfg.get('engine', {}).get('backend')
    if backend == 'cuda_legacy':
        errors = validate_cuda_legacy_compatibility(cfg)
    elif backend == 'cuda_native':
        errors = validate_cuda_native_config(cfg)
    else:
        errors = validate_backend_model_capabilities(cfg.get('model', {}), backend)
    payload = {
        'schema_version': 1,
        'kind': 'validation_result',
        'ok': not errors,
        'status': 'ok' if not errors else 'error',
        'errors': errors,
        'backend': backend,
    }
    _print_validation_result(payload, command='validate-config', output_format=args.format)
    return 0 if not errors else 2


def handle_validate_cuda_native_config(args) -> int:
    from minicnn.cuda_native.api import validate_cuda_native_config

    cfg = _load_unified_config_or_exit(args.config, args.overrides)
    errors = validate_cuda_native_config(cfg)
    payload = {
        'schema_version': 1,
        'kind': 'validation_result',
        'ok': not errors,
        'status': 'ok' if not errors else 'error',
        'errors': errors,
        'backend': 'cuda_native',
    }
    if not errors:
        payload['note'] = 'experimental — backward/training prototypes present, strict boundary validation applied'
    _print_validation_result(payload, command='validate-cuda-native-config', output_format=args.format)
    return 0 if not errors else 2


def handle_cuda_native_capabilities() -> int:
    from minicnn.cuda_native.api import get_capability_summary as get_cuda_native_summary

    _print_json(get_cuda_native_summary())
    return 0
