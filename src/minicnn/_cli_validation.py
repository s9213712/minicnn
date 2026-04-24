from __future__ import annotations

from minicnn._cli_config import _load_unified_config_or_exit
from minicnn._cli_output import _print_json, _print_validation_result
from minicnn.backend_capability import validate_backend_model_capabilities


def handle_validate_config(args) -> int:
    from minicnn.cuda_native.api import (
        assess_cuda_native_execution_readiness,
        assess_cuda_native_support_tier,
        resolve_cuda_native_execution_mode,
        validate_cuda_native_config,
    )
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
        'schema_name': 'minicnn.cli.validation',
        'schema_version': 1,
        'artifact_kind': 'validation_result',
        'kind': 'validation_result',
        'ok': not errors,
        'status': 'ok' if not errors else 'error',
        'errors': errors,
        'backend': backend,
    }
    if backend == 'cuda_native':
        payload['support_tier_assessment'] = assess_cuda_native_support_tier(cfg)
        payload['execution_readiness_assessment'] = assess_cuda_native_execution_readiness(cfg)
        payload.update(resolve_cuda_native_execution_mode(cfg))
    _print_validation_result(payload, command='validate-config', output_format=args.format)
    return 0 if not errors else 2


def handle_validate_cuda_native_config(args) -> int:
    from minicnn.cuda_native.api import (
        assess_cuda_native_execution_readiness,
        assess_cuda_native_support_tier,
        resolve_cuda_native_execution_mode,
        validate_cuda_native_config,
    )

    cfg = _load_unified_config_or_exit(args.config, args.overrides)
    errors = validate_cuda_native_config(cfg)
    payload = {
        'schema_name': 'minicnn.cli.validation',
        'schema_version': 1,
        'artifact_kind': 'validation_result',
        'kind': 'validation_result',
        'ok': not errors,
        'status': 'ok' if not errors else 'error',
        'errors': errors,
        'backend': 'cuda_native',
        'support_tier_assessment': assess_cuda_native_support_tier(cfg),
        'execution_readiness_assessment': assess_cuda_native_execution_readiness(cfg),
    }
    payload.update(resolve_cuda_native_execution_mode(cfg))
    if not errors:
        payload['note'] = (
            'beta-grade backend; reference_numpy is broadly supported, while gpu_native is '
            'supported for the narrow Linear, Linear+ReLU, MaxPool+Linear, '
            'and Conv2d(valid, bias=false)+Linear / CrossEntropyLoss / SGD training subsets'
        )
    _print_validation_result(payload, command='validate-cuda-native-config', output_format=args.format)
    return 0 if not errors else 2


def handle_cuda_native_capabilities() -> int:
    from minicnn.cuda_native.api import get_capability_summary as get_cuda_native_summary

    _print_json(get_cuda_native_summary())
    return 0
