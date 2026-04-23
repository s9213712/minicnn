from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minicnn._cli_config import _load_flex_config_or_exit, _load_unified_config_or_exit, _resolve_cli_config_path
from minicnn._cli_errors import _exit_user_error
from minicnn._cli_output import (
    _print_diagnostic,
    _print_generic_payload,
    _print_graph_view,
    _print_json,
    _print_model_view,
    _print_validation_result,
)
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT


def _smoke_check(
    name: str,
    ok: bool,
    *,
    required: bool = True,
    details: dict[str, Any] | None = None,
    suggested_fix: str | None = None,
) -> dict[str, Any]:
    return {
        'name': name,
        'ok': bool(ok),
        'required': required,
        'severity': 'info' if ok else ('error' if required else 'warning'),
        'details': details or {},
        'suggested_fix': suggested_fix or '',
    }


def run_smoke_checks() -> dict[str, Any]:
    from minicnn.compiler import optimize, trace_model_config
    from minicnn.cuda_native.api import validate_cuda_native_config
    from minicnn.flex.config import load_flex_config
    from minicnn.framework.health import build_diagnostic_payload, healthcheck
    from minicnn.unified.config import load_unified_config
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility

    health = healthcheck()
    checks: list[dict[str, Any]] = []

    checks.append(_smoke_check(
        'project_paths',
        bool(health.get('project_root_exists')) and bool(health.get('cpp_root_exists')),
        details={
            'project_root': str(PROJECT_ROOT),
            'cpp_root': str(CPP_ROOT),
        },
    ))

    flex_registries = health.get('flex_registries', {})
    checks.append(_smoke_check(
        'flex_registry_surface',
        bool(flex_registries.get('layers')) and bool(flex_registries.get('optimizers')),
        details={'registries': flex_registries},
    ))

    native_artifacts = list(health.get('native_artifacts', health.get('shared_objects', [])))
    checks.append(_smoke_check(
        'native_cuda_artifacts',
        bool(native_artifacts),
        required=False,
        details={
            'native_artifacts': native_artifacts,
            'shared_objects': native_artifacts,
            'hint': 'Run minicnn build --legacy-make --check if you need cuda_legacy.',
        },
        suggested_fix='Run minicnn build --legacy-make --check if you need cuda_legacy.',
    ))

    cifar10_ready = bool(health.get('data_root_exists'))
    checks.append(_smoke_check(
        'cifar10_data',
        cifar10_ready,
        required=False,
        details={
            'data_root': str(DATA_ROOT),
            'hint': 'Run minicnn prepare-data if you want the handcrafted CUDA CIFAR-10 path.',
        },
        suggested_fix='Run minicnn prepare-data if you want the handcrafted CUDA CIFAR-10 path.',
    ))

    flex_config_path = _resolve_cli_config_path('configs/flex_cnn.yaml')
    flex_cfg = load_flex_config(flex_config_path)
    checks.append(_smoke_check(
        'flex_config_parse',
        True,
        details={'config': flex_config_path},
    ))

    graph = optimize(trace_model_config(flex_cfg.get('model', {})))
    checks.append(_smoke_check(
        'compiler_trace',
        True,
        details={
            'config': flex_config_path,
            'summary': graph.summary(),
        },
    ))

    legacy_config_path = _resolve_cli_config_path('configs/cuda_legacy_strict.yaml')
    legacy_cfg = load_unified_config(legacy_config_path)
    legacy_errors = validate_cuda_legacy_compatibility(legacy_cfg)
    checks.append(_smoke_check(
        'cuda_legacy_validation',
        not legacy_errors,
        details={
            'config': legacy_config_path,
            'errors': legacy_errors,
        },
    ))

    native_config_path = _resolve_cli_config_path('configs/dual_backend_cnn.yaml')
    native_cfg = load_unified_config(native_config_path, ['engine.backend=cuda_native'])
    native_errors = validate_cuda_native_config(native_cfg)
    checks.append(_smoke_check(
        'cuda_native_validation',
        not native_errors,
        details={
            'config': native_config_path,
            'errors': native_errors,
        },
    ))

    overall_ok = all(check['ok'] for check in checks if check['required'])
    next_steps: list[str] = []
    if not native_artifacts:
        next_steps.append('minicnn build --legacy-make --check')
    if not cifar10_ready:
        next_steps.append('minicnn prepare-data')
    if overall_ok:
        next_steps.append('minicnn train-flex --config configs/flex_cnn.yaml')

    return build_diagnostic_payload(checks=checks, extra={
        'ok': overall_ok,
        'next_steps': next_steps,
    })


def handle_info(args) -> int:
    from minicnn.config import settings
    from minicnn.core.cuda_backend import resolve_library_path
    from minicnn.flex.registry import describe_registries
    from minicnn.framework.health import healthcheck
    from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED

    cuda_library = resolve_library_path()
    payload = {
        'schema_version': 1,
        'kind': 'project_info',
        'status': 'ok',
        'project_root': str(PROJECT_ROOT),
        'cpp_root': str(CPP_ROOT),
        'data_root': str(DATA_ROOT),
        'cuda_library': str(cuda_library),
        'native_library_present': Path(cuda_library).exists(),
        'resolved_legacy_settings': settings.summarize(),
        'health': healthcheck(),
        'flexible_registries': describe_registries(),
        'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED,
    }
    if args.format == 'json':
        _print_json({'command': 'info', **payload})
        return 0
    print(f'PROJECT_ROOT={PROJECT_ROOT}')
    print(f'CPP_ROOT={CPP_ROOT}')
    print(f'DATA_ROOT={DATA_ROOT}')
    print(f'CUDA_LIBRARY={cuda_library}')
    print(f'Native library present={Path(cuda_library).exists()}')
    print('Resolved legacy settings:')
    print(json.dumps(settings.summarize(), indent=2))
    print('Health:')
    _print_json(healthcheck())
    print('Flexible registries:')
    _print_json(describe_registries())
    print('Dual-backend handcrafted CUDA supported subset:')
    _print_json(CUDA_LEGACY_SUPPORTED)
    return 0


def handle_healthcheck(args) -> int:
    from minicnn.framework.health import healthcheck

    _print_diagnostic(healthcheck(), command='healthcheck', output_format=args.format)
    return 0


def handle_smoke(args) -> int:
    result = run_smoke_checks()
    _print_diagnostic(result, command='smoke', output_format=args.format)
    return 0 if result['ok'] else 2


def handle_doctor(args) -> int:
    from minicnn.framework.health import doctor

    _print_diagnostic(doctor(), command='doctor', output_format=args.format)
    return 0


def handle_list_flex_components() -> int:
    from minicnn.flex.registry import describe_registries

    _print_json({
        'schema_version': 1,
        'kind': 'flex_component_registry',
        'status': 'ok',
        'registries': describe_registries(),
    })
    return 0


def handle_list_dual_components() -> int:
    from minicnn.cuda_native.api import get_capability_summary as get_cuda_native_summary
    from minicnn.flex.registry import describe_registries
    from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED

    _print_json({
        'schema_version': 1,
        'kind': 'dual_component_registry',
        'status': 'ok',
        'registries': describe_registries(),
        'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED,
        'cuda_native_capabilities': get_cuda_native_summary(),
    })
    return 0


def handle_config_template() -> int:
    from minicnn.flex.config import dump_template

    print(dump_template())
    return 0


def handle_dual_config_template() -> int:
    from minicnn.unified.config import dump_unified_template

    print(dump_unified_template())
    return 0


def handle_inspect_checkpoint(args) -> int:
    from minicnn.artifacts import inspect_checkpoint

    try:
        payload = inspect_checkpoint(args.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _exit_user_error(str(exc))
    _print_generic_payload(payload, command='inspect-checkpoint', output_format=args.format)
    return 0


def handle_export_torch_checkpoint(args) -> int:
    from minicnn.artifacts import export_checkpoint_to_torch

    try:
        payload = export_checkpoint_to_torch(
            args.path,
            config_path=args.config,
            output_path=args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        _exit_user_error(str(exc))
    _print_json(payload)
    return 0


def handle_evaluate_checkpoint(args) -> int:
    from minicnn.inference import evaluate_checkpoint, resolve_checkpoint_path

    cfg = _load_unified_config_or_exit(args.config, args.overrides)
    try:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_path=args.checkpoint,
            summary_path=args.summary,
        )
        payload = evaluate_checkpoint(
            cfg,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
            batch_size=args.batch_size,
            test_data_path=args.test_data,
            test_data_normalized=bool(args.test_data_normalized),
        )
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        _exit_user_error(str(exc))

    if args.format == 'json':
        _print_json({'command': 'evaluate-checkpoint', **payload})
        return 0

    print(f"evaluate-checkpoint: {payload['status']}")
    print(f"checkpoint_path: {payload['checkpoint_path']}")
    print(f"dataset_source: {payload['dataset_source']}")
    print(f"device: {payload['device']}")
    print(f"num_samples: {payload['num_samples']}")
    print(f"batch_size: {payload['batch_size']}")
    print(f"loss_type: {payload['loss_type']}")
    print(f"accuracy: {payload['accuracy'] * 100:.2f}%")
    print(f"loss: {payload['loss']:.6f}")
    return 0


def handle_validate_dual_config(args) -> int:
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility

    cfg = _load_unified_config_or_exit(args.config, args.overrides)
    errors = validate_cuda_legacy_compatibility(cfg)
    payload = {
        'schema_version': 1,
        'kind': 'validation_result',
        'ok': not errors,
        'status': 'ok' if not errors else 'error',
        'backend': 'cuda_legacy',
        'errors': errors,
    }
    _print_validation_result(payload, command='validate-dual-config', output_format=args.format)
    return 0 if not errors else 2


def handle_show_cuda_mapping(args) -> int:
    from minicnn.unified.cuda_legacy import summarize_legacy_mapping

    cfg = _load_unified_config_or_exit(args.config, args.overrides)
    payload = {
        'schema_version': 1,
        'kind': 'cuda_legacy_mapping',
        'status': 'ok',
        **summarize_legacy_mapping(cfg),
    }
    _print_generic_payload(
        payload,
        command='show-cuda-mapping',
        output_format=args.format,
    )
    return 0


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
        errors = []
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


def handle_compile(args) -> int:
    from minicnn.compiler import optimize, trace_model_config

    cfg = _load_flex_config_or_exit(args.config, args.overrides)
    graph = optimize(trace_model_config(cfg.get('model', {})))
    _print_json({
        'command': 'compile',
        'schema_version': 1,
        'kind': 'compiled_graph_summary',
        'status': 'ok',
        **graph.summary(),
    })
    return 0


def handle_show_model(args) -> int:
    from minicnn.introspection.model_view import build_model_view_from_config, render_model_view_text

    cfg = _load_flex_config_or_exit(args.config, args.overrides)
    view = build_model_view_from_config(cfg)
    payload = {
        'status': 'ok',
        'schema_version': 1,
        'kind': 'model_view',
        'model_type': view.model_type,
        'input_shape': view.input_shape,
        'backend_intent': view.backend_intent,
        'summary': view.summary,
        'layers': [layer.to_dict() for layer in view.layers],
        'text': render_model_view_text(view),
    }
    _print_model_view(payload, command='show-model', output_format=args.format)
    return 0


def handle_show_graph(args) -> int:
    from minicnn.introspection.graph_view import build_graph_view_from_config, render_graph_view_text

    cfg = _load_flex_config_or_exit(args.config, args.overrides)
    payload = {
        'status': 'ok',
        'schema_version': 1,
        'kind': 'graph_view',
        **build_graph_view_from_config(cfg),
    }
    payload['text'] = render_graph_view_text(payload)
    _print_graph_view(payload, command='show-graph', output_format=args.format)
    return 0
