from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

from minicnn.core.build import build_native, check_native
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT


_COMPARE_BACKENDS = {'torch', 'cuda_legacy', 'autograd'}
_TORCH_INSTALL_HINT = 'Install it with:\n  pip install -e .[torch]'


def _exit_user_error(message: str) -> None:
    print(message)
    raise SystemExit(2)


def _ensure_torch_or_exit(command_name: str) -> None:
    try:
        importlib.import_module('torch')
    except Exception:
        _exit_user_error(f'{command_name} requires PyTorch.\n{_TORCH_INSTALL_HINT}')


def _ensure_cuda_legacy_prereqs_or_exit(cfg: dict[str, Any]) -> None:
    from minicnn.core.cuda_backend import resolve_library_path
    from minicnn.data.cifar10 import cifar10_ready

    library_path = Path(resolve_library_path())
    if not library_path.exists():
        _exit_user_error(
            'cuda_legacy training requires a native CUDA shared library.\n'
            'Build it with:\n'
            '  minicnn build --legacy-make --check'
        )
    dataset_cfg = cfg.get('dataset', {})
    if str(dataset_cfg.get('type', 'cifar10')) == 'cifar10':
        data_root = Path(dataset_cfg.get('data_root', DATA_ROOT))
        if not cifar10_ready(data_root):
            _exit_user_error(
                'cuda_legacy training requires prepared CIFAR-10 data.\n'
                'Prepare it with:\n'
                '  minicnn prepare-data'
            )


def _resolve_cli_config_path(config_path: str | None) -> str | None:
    if not config_path:
        return config_path
    raw_path = Path(config_path)
    if raw_path.exists():
        return str(raw_path)
    if not raw_path.is_absolute():
        project_relative = PROJECT_ROOT / raw_path
        if project_relative.exists():
            return str(project_relative)
    return config_path


def _load_flex_config_or_exit(config_path: str, overrides: list[str]) -> dict:
    from minicnn.flex.config import load_flex_config

    resolved_path = _resolve_cli_config_path(config_path)
    try:
        return load_flex_config(resolved_path, overrides)
    except FileNotFoundError:
        _exit_user_error(
            f"Config file not found: {config_path}\n"
            "Create a template with:\n"
            "  minicnn config-template > configs/my_config.yaml"
        )


def _load_unified_config_or_exit(config_path: str, overrides: list[str]) -> dict:
    from minicnn.unified.config import load_unified_config

    resolved_path = _resolve_cli_config_path(config_path)
    try:
        return load_unified_config(resolved_path, overrides)
    except FileNotFoundError:
        _exit_user_error(
            f"Config file not found: {config_path}\n"
            "Create a template with:\n"
            "  minicnn dual-config-template > configs/my_config.yaml"
        )


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--lr-conv1', type=float)
    parser.add_argument('--lr-conv', type=float)
    parser.add_argument('--lr-fc', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--dataset-seed', type=int)
    parser.add_argument('--init-seed', type=int)
    parser.add_argument('--train-seed', type=int)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--eval-max-batches', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--grad-debug', action='store_true')
    parser.add_argument('--grad-debug-batches', type=int)


def _common_train_overrides(args) -> list[str]:
    mapping = {
        'epochs': 'train.epochs',
        'batch': 'train.batch_size',
        'lr_conv1': 'optimizer.lr_conv1',
        'lr_conv': 'optimizer.lr_conv',
        'lr_fc': 'optimizer.lr_fc',
        'momentum': 'optimizer.momentum',
        'weight_decay': 'optimizer.weight_decay',
        'dataset_seed': 'dataset.seed',
        'train_seed': 'train.seed',
        'data_dir': 'dataset.data_root',
        'eval_max_batches': 'train.eval_max_batches',
        'log_interval': 'train.log_every',
        'grad_debug_batches': 'runtime.grad_debug_batches',
    }
    overrides = []
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides.append(f'{key}={value}')
    if getattr(args, 'init_seed', None) is not None:
        overrides.append(f'train.init_seed={args.init_seed}')
    if getattr(args, 'checkpoint_path', None) is not None:
        overrides.append(f'runtime.best_model_filename={args.checkpoint_path}')
    if getattr(args, 'grad_debug', False):
        overrides.append('runtime.grad_debug=true')
    return overrides


def _read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / 'metrics.jsonl'
    if not metrics_path.exists():
        return []
    rows = []
    for line in metrics_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _effective_train_samples(cfg: dict[str, Any]) -> int:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    num_samples = int(dataset_cfg.get('num_samples', 0) or 0)
    batch_size = int(train_cfg.get('batch_size', 0) or 0)
    max_steps = train_cfg.get('max_steps_per_epoch')
    if max_steps is not None and batch_size > 0:
        capped = int(max_steps) * batch_size
        return min(num_samples, capped) if num_samples > 0 else capped
    return num_samples


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def _benchmark_fields(backend: str, cfg: dict[str, Any], run_dir: Path, elapsed_s: float) -> dict[str, Any]:
    train_cfg = cfg.get('train', {})
    dataset_cfg = cfg.get('dataset', {})
    runtime_cfg = cfg.get('runtime', {})
    metrics = _read_metrics(run_dir)
    epoch_times = [
        float(row['epoch_time_s'])
        for row in metrics
        if row.get('epoch_time_s') is not None
    ]
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else None
    train_samples = _effective_train_samples(cfg)
    throughput_window = avg_epoch_time if avg_epoch_time and avg_epoch_time > 0 else elapsed_s
    samples_per_sec = (
        train_samples / throughput_window
        if train_samples > 0 and throughput_window > 0
        else None
    )
    if backend == 'cuda_legacy':
        variant = runtime_cfg.get('cuda_variant')
    elif backend == 'torch':
        variant = train_cfg.get('device')
    else:
        variant = ''
    return {
        'variant': variant or '',
        'train_samples': train_samples,
        'val_samples': int(dataset_cfg.get('val_samples', 0) or 0),
        'batch_size': int(train_cfg.get('batch_size', 0) or 0),
        'epochs_requested': int(train_cfg.get('epochs', 0) or 0),
        'epochs_completed': len(metrics) if metrics else None,
        'avg_epoch_time_s': _round_or_none(avg_epoch_time, digits=6),
        'last_epoch_time_s': _round_or_none(epoch_times[-1] if epoch_times else None, digits=6),
        'samples_per_sec': _round_or_none(samples_per_sec),
        'elapsed_s': round(elapsed_s, 3),
    }


def _compare_backends_and_overrides(args) -> tuple[list[str], list[str]]:
    if args.backends:
        backends = []
        extra_overrides = []
        for token in args.backends:
            if token in _COMPARE_BACKENDS and not extra_overrides:
                backends.append(token)
            elif '=' in token:
                extra_overrides.append(token)
            else:
                expected = ', '.join(sorted(_COMPARE_BACKENDS))
                raise ValueError(f"invalid compare backend {token!r}; expected one of: {expected}")
        return backends, [*args.overrides, *extra_overrides]
    backends = [b for b in (args.backend_a, args.backend_b) if b] or ['torch', 'cuda_legacy']
    return backends, args.overrides


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


def _run_smoke_checks() -> dict[str, Any]:
    from minicnn.compiler import optimize, trace_model_config
    from minicnn.cuda_native.api import validate_cuda_native_config
    from minicnn.flex.config import load_flex_config
    from minicnn.framework.health import healthcheck
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

    shared_objects = list(health.get('shared_objects', []))
    checks.append(_smoke_check(
        'native_cuda_artifacts',
        bool(shared_objects),
        required=False,
        details={
            'shared_objects': shared_objects,
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
    if not shared_objects:
        next_steps.append('minicnn build --legacy-make --check')
    if not cifar10_ready:
        next_steps.append('minicnn prepare-data')
    if overall_ok:
        next_steps.append('minicnn train-flex --config configs/flex_cnn.yaml')

    return {
        'ok': overall_ok,
        'checks': checks,
        'next_steps': next_steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='minicnn', description='MiniCNN dual-backend CLI (pure handcrafted CUDA + PyTorch)')
    sub = parser.add_subparsers(dest='command', required=True)

    p_build = sub.add_parser('build', help='Build CUDA shared library')
    p_build.add_argument('--no-cublas', action='store_true', help='Disable cuBLAS and use handwritten fallback')
    p_build.add_argument('--check', action='store_true', help='Run symbol/export checks after build')
    p_build.add_argument('--generator', choices=['make', 'ninja'], default='make')
    p_build.add_argument('--legacy-make', action='store_true', help='Use original cpp/Makefile instead of CMake')
    p_build.add_argument('--cuda-arch', default='86', help='CUDA architecture, for example 86 or sm_86')
    p_build.add_argument(
        '--variant',
        choices=['default', 'cublas', 'handmade', 'both'],
        default='default',
        help='Native library output variant to build',
    )

    sub.add_parser('prepare-data', help='Download and extract CIFAR-10 Python batches')
    sub.add_parser('info', help='Show important project paths and summary')
    sub.add_parser('doctor', help='Run first-run diagnostics for paths, data, and native CUDA')
    sub.add_parser('healthcheck', help='Validate framework wiring and native artifacts')
    sub.add_parser('smoke', help='Run a compact first-run self-check across configs, compiler, and backend validation')
    sub.add_parser('list-flex-components', help='List configurable built-in components')
    sub.add_parser('list-dual-components', help='List dual-backend components and cuda_legacy subset')
    sub.add_parser('config-template', help='Print the PyTorch-flex config template')
    sub.add_parser('dual-config-template', help='Print the dual-backend unified config template')
    p_inspect_ckpt = sub.add_parser('inspect-checkpoint', help='Inspect a saved model/checkpoint artifact (.pt/.pth/.npz)')
    p_inspect_ckpt.add_argument('--path', required=True, type=str)
    p_export_ckpt = sub.add_parser('export-torch-checkpoint', help='Export a supported MiniCNN checkpoint to a generic torch checkpoint (.pt)')
    p_export_ckpt.add_argument('--path', required=True, type=str)
    p_export_ckpt.add_argument('--config', required=True, type=str)
    p_export_ckpt.add_argument('--output', required=True, type=str)

    p_flex = sub.add_parser('train-flex', help='Train a configurable PyTorch model from YAML')
    p_flex.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    _add_common_train_args(p_flex)
    p_flex.add_argument('overrides', nargs='*', help='Overrides like train.epochs=2 optimizer.lr=0.001')

    p_dual = sub.add_parser('train-dual', help='Train with one shared model config and switch backend via engine.backend')
    p_dual.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_dual)
    p_dual.add_argument('overrides', nargs='*', help='Overrides like engine.backend=cuda_legacy train.epochs=2')

    p_train = sub.add_parser('train', help='Alias for train-dual')
    p_train.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_train)
    p_train.add_argument('overrides', nargs='*')

    p_train_cuda = sub.add_parser('train-cuda', help='Alias for train-dual with engine.backend=cuda_legacy')
    p_train_cuda.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_train_cuda)
    p_train_cuda.add_argument('overrides', nargs='*')

    p_train_torch = sub.add_parser('train-torch', help='Alias for train-dual with engine.backend=torch')
    p_train_torch.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    _add_common_train_args(p_train_torch)
    p_train_torch.add_argument('overrides', nargs='*')

    p_train_autograd = sub.add_parser('train-autograd', help='Train the CPU/NumPy MiniCNN autograd path')
    p_train_autograd.add_argument('--config', type=str, default='configs/autograd_tiny.yaml')
    _add_common_train_args(p_train_autograd)
    p_train_autograd.add_argument('overrides', nargs='*')

    p_compare = sub.add_parser('compare', help='Run backend comparison with one shared config')
    p_compare.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_compare.add_argument('--backends', nargs='+', default=None, metavar='BACKEND')
    p_compare.add_argument('--backend-a', choices=['torch', 'cuda_legacy', 'autograd'])
    p_compare.add_argument('--backend-b', choices=['torch', 'cuda_legacy', 'autograd'])
    _add_common_train_args(p_compare)
    p_compare.add_argument('overrides', nargs='*')

    p_validate = sub.add_parser('validate-dual-config', help='Validate whether a config can run on cuda_legacy')
    p_validate.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate.add_argument('overrides', nargs='*')

    p_map = sub.add_parser('show-cuda-mapping', help='Show how a unified config maps onto the handcrafted CUDA backend')
    p_map.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_map.add_argument('overrides', nargs='*')

    p_validate_any = sub.add_parser('validate-config', help='Validate shared config shape and backend compatibility')
    p_validate_any.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate_any.add_argument('overrides', nargs='*')

    p_compile = sub.add_parser('compile', help='Trace and optimize a model config into MiniCNN IR')
    p_compile.add_argument('--config', type=str, default='configs/autograd_tiny.yaml')
    p_compile.add_argument('overrides', nargs='*')

    p_train_native = sub.add_parser('train-native', help='[EXPERIMENTAL] Train with cuda_native backend (research prototype)')
    p_train_native.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_train_native.add_argument('overrides', nargs='*')

    p_validate_native = sub.add_parser('validate-cuda-native-config', help='[EXPERIMENTAL] Validate config against cuda_native constraints')
    p_validate_native.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate_native.add_argument('overrides', nargs='*')

    sub.add_parser('cuda-native-capabilities', help='[EXPERIMENTAL] Print cuda_native capability descriptor')

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'build':
        build_native(
            use_cublas=not args.no_cublas,
            generator=args.generator,
            legacy_make=args.legacy_make,
            variant=args.variant,
            cuda_arch=args.cuda_arch,
        )
        if args.check:
            check_native(args.variant)
        return 0

    if args.command == 'prepare-data':
        from minicnn.data.prepare_cifar10 import main as prepare_data_main
        prepare_data_main()
        return 0

    if args.command == 'info':
        from minicnn.config import settings
        from minicnn.core.cuda_backend import resolve_library_path
        from minicnn.flex.registry import describe_registries
        from minicnn.framework.health import healthcheck
        from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED

        cuda_library = resolve_library_path()
        print(f'PROJECT_ROOT={PROJECT_ROOT}')
        print(f'CPP_ROOT={CPP_ROOT}')
        print(f'DATA_ROOT={DATA_ROOT}')
        print(f'CUDA_LIBRARY={cuda_library}')
        print(f'Native library present={Path(cuda_library).exists()}')
        print('Resolved legacy settings:')
        print(json.dumps(settings.summarize(), indent=2))
        print(f'Health={healthcheck()}')
        print('Flexible registries:')
        print(json.dumps(describe_registries(), indent=2))
        print('Dual-backend handcrafted CUDA supported subset:')
        print(json.dumps(CUDA_LEGACY_SUPPORTED, indent=2))
        return 0

    if args.command == 'healthcheck':
        from minicnn.framework.health import healthcheck

        print(healthcheck())
        return 0

    if args.command == 'smoke':
        result = _run_smoke_checks()
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 2

    if args.command == 'doctor':
        from minicnn.framework.health import doctor

        print(json.dumps(doctor(), indent=2, default=str))
        return 0

    if args.command == 'list-flex-components':
        from minicnn.flex.registry import describe_registries

        print(json.dumps(describe_registries(), indent=2))
        return 0

    if args.command == 'list-dual-components':
        from minicnn.cuda_native.api import get_capability_summary as get_cuda_native_summary
        from minicnn.flex.registry import describe_registries
        from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED

        print(json.dumps({
            'registries': describe_registries(),
            'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED,
            'cuda_native_capabilities': get_cuda_native_summary(),
        }, indent=2))
        return 0

    if args.command == 'config-template':
        from minicnn.flex.config import dump_template

        print(dump_template())
        return 0

    if args.command == 'dual-config-template':
        from minicnn.unified.config import dump_unified_template

        print(dump_unified_template())
        return 0

    if args.command == 'inspect-checkpoint':
        from minicnn.artifacts import inspect_checkpoint

        try:
            payload = inspect_checkpoint(args.path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            _exit_user_error(str(exc))
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == 'export-torch-checkpoint':
        from minicnn.artifacts import export_checkpoint_to_torch

        try:
            payload = export_checkpoint_to_torch(
                args.path,
                config_path=args.config,
                output_path=args.output,
            )
        except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
            _exit_user_error(str(exc))
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == 'train-flex':
        _ensure_torch_or_exit('train-flex')
        from minicnn.flex.trainer import train_from_config

        cfg = _load_flex_config_or_exit(args.config, [*_common_train_overrides(args), *args.overrides])
        run_dir = train_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'validate-dual-config':
        from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility

        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        errors = validate_cuda_legacy_compatibility(cfg)
        if errors:
            print(json.dumps({'ok': False, 'errors': errors}, indent=2))
            return 2
        print(json.dumps({'ok': True, 'backend': 'cuda_legacy'}, indent=2))
        return 0

    if args.command == 'show-cuda-mapping':
        from minicnn.unified.cuda_legacy import summarize_legacy_mapping

        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        print(json.dumps(summarize_legacy_mapping(cfg), indent=2))
        return 0

    if args.command in {'train', 'train-dual'}:
        cfg = _load_unified_config_or_exit(args.config, [*_common_train_overrides(args), *args.overrides])
        backend = str(cfg.get('engine', {}).get('backend', 'torch'))
        if backend == 'torch':
            _ensure_torch_or_exit('train-dual with engine.backend=torch')
        elif backend == 'cuda_legacy':
            _ensure_cuda_legacy_prereqs_or_exit(cfg)
        from minicnn.unified.trainer import train_unified_from_config

        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command in {'train-cuda', 'train-torch'}:
        backend = 'cuda_legacy' if args.command == 'train-cuda' else 'torch'
        cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *_common_train_overrides(args), *args.overrides])
        if backend == 'torch':
            _ensure_torch_or_exit('train-dual with engine.backend=torch')
        else:
            _ensure_cuda_legacy_prereqs_or_exit(cfg)
        from minicnn.unified.trainer import train_unified_from_config

        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'train-autograd':
        from minicnn.training.train_autograd import train_autograd_from_config
        cfg = _load_flex_config_or_exit(args.config if args.config else None, [*_common_train_overrides(args), *args.overrides])
        run_dir = train_autograd_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'compare':
        rows = []
        try:
            backends, compare_overrides = _compare_backends_and_overrides(args)
        except ValueError as exc:
            parser.error(str(exc))
        for backend in backends:
            t0 = time.perf_counter()
            if backend == 'autograd':
                from minicnn.training.train_autograd import train_autograd_from_config
                cfg = _load_flex_config_or_exit(args.config if args.config else None, [*_common_train_overrides(args), *compare_overrides])
                run_dir = train_autograd_from_config(cfg)
            else:
                cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *_common_train_overrides(args), *compare_overrides])
                if backend == 'torch':
                    _ensure_torch_or_exit('compare with engine.backend=torch')
                elif backend == 'cuda_legacy':
                    _ensure_cuda_legacy_prereqs_or_exit(cfg)
                from minicnn.unified.trainer import train_unified_from_config
                run_dir = train_unified_from_config(cfg)
            elapsed = time.perf_counter() - t0
            summary_path = run_dir / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
            row = {
                'backend': backend,
                'run_dir': str(run_dir),
                'best_model_path': summary.get('best_model_path'),
                'test_acc': summary.get('test_acc'),
            }
            row.update(_benchmark_fields(backend, cfg, run_dir, elapsed))
            rows.append(row)
        print(json.dumps({'runs': rows}, indent=2))
        return 0

    if args.command == 'validate-config':
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
        print(json.dumps({'ok': not errors, 'errors': errors, 'backend': backend}, indent=2))
        return 0 if not errors else 2

    if args.command == 'validate-cuda-native-config':
        from minicnn.cuda_native.api import validate_cuda_native_config

        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        errors = validate_cuda_native_config(cfg)
        if errors:
            print(json.dumps({'ok': False, 'errors': errors, 'backend': 'cuda_native'}, indent=2))
            return 2
        print(json.dumps({
            'ok': True,
            'backend': 'cuda_native',
            'note': 'experimental — backward/training prototypes present, strict boundary validation applied',
        }, indent=2))
        return 0

    if args.command == 'cuda-native-capabilities':
        from minicnn.cuda_native.api import get_capability_summary as get_cuda_native_summary

        print(json.dumps(get_cuda_native_summary(), indent=2))
        return 0

    if args.command == 'train-native':
        import warnings
        from minicnn.cuda_native.api import get_capability_summary as get_cuda_native_summary
        from minicnn.unified.trainer import train_unified_from_config

        cfg = _load_unified_config_or_exit(args.config, ['engine.backend=cuda_native', *_common_train_overrides(args), *args.overrides])
        warnings.warn(
            '[EXPERIMENTAL] cuda_native backend: backward/training prototypes exist, '
            'but the validated support boundary remains narrow and not production-ready.',
            stacklevel=1,
        )
        summary = get_cuda_native_summary()
        print(json.dumps({
            'backend': 'cuda_native',
            'status': 'experimental',
            'validated_support_boundary': {
                'datasets': summary.get('supported_datasets', []),
                'losses': summary.get('supported_losses', []),
                'optimizers': summary.get('supported_optimizers', []),
                'schedulers': summary.get('supported_schedulers', []),
                'ops': summary.get('supported_ops', []),
            },
        }, indent=2))
        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'compile':
        from minicnn.compiler import optimize, trace_model_config
        cfg = _load_flex_config_or_exit(args.config, args.overrides)
        graph = optimize(trace_model_config(cfg.get('model', {})))
        print(json.dumps(graph.summary(), indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
