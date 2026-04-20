from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from minicnn.core.build import build_native, check_native
from minicnn.flex.config import load_flex_config, dump_template
from minicnn.flex.registry import describe_registries
from minicnn.flex.trainer import train_from_config
from minicnn.framework.health import doctor, healthcheck
from minicnn.paths import CPP_ROOT, DATA_ROOT, PROJECT_ROOT
from minicnn.unified.config import load_unified_config, dump_unified_template
from minicnn.unified.cuda_legacy import CUDA_LEGACY_SUPPORTED, summarize_legacy_mapping, validate_cuda_legacy_compatibility
from minicnn.unified.trainer import train_unified_from_config


_COMPARE_BACKENDS = {'torch', 'cuda_legacy', 'autograd'}


def _load_flex_config_or_exit(config_path: str, overrides: list[str]) -> dict:
    try:
        return load_flex_config(config_path, overrides)
    except FileNotFoundError:
        print(
            f"Config file not found: {config_path}\n"
            "Create a template with:\n"
            "  minicnn config-template > configs/my_config.yaml"
        )
        raise SystemExit(2)


def _load_unified_config_or_exit(config_path: str, overrides: list[str]) -> dict:
    try:
        return load_unified_config(config_path, overrides)
    except FileNotFoundError:
        print(
            f"Config file not found: {config_path}\n"
            "Create a template with:\n"
            "  minicnn dual-config-template > configs/my_config.yaml"
        )
        raise SystemExit(2)


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
    sub.add_parser('list-flex-components', help='List configurable built-in components')
    sub.add_parser('list-dual-components', help='List dual-backend components and cuda_legacy subset')
    sub.add_parser('config-template', help='Print the PyTorch-flex config template')
    sub.add_parser('dual-config-template', help='Print the dual-backend unified config template')

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
        print(healthcheck())
        return 0

    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=2, default=str))
        return 0

    if args.command == 'list-flex-components':
        print(json.dumps(describe_registries(), indent=2))
        return 0

    if args.command == 'list-dual-components':
        print(json.dumps({'registries': describe_registries(), 'cuda_legacy_subset': CUDA_LEGACY_SUPPORTED}, indent=2))
        return 0

    if args.command == 'config-template':
        print(dump_template())
        return 0

    if args.command == 'dual-config-template':
        print(dump_unified_template())
        return 0

    if args.command == 'train-flex':
        cfg = _load_flex_config_or_exit(args.config, [*_common_train_overrides(args), *args.overrides])
        run_dir = train_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'validate-dual-config':
        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        errors = validate_cuda_legacy_compatibility(cfg)
        if errors:
            print(json.dumps({'ok': False, 'errors': errors}, indent=2))
            return 2
        print(json.dumps({'ok': True, 'backend': 'cuda_legacy'}, indent=2))
        return 0

    if args.command == 'show-cuda-mapping':
        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        print(json.dumps(summarize_legacy_mapping(cfg), indent=2))
        return 0

    if args.command in {'train', 'train-dual'}:
        cfg = _load_unified_config_or_exit(args.config, [*_common_train_overrides(args), *args.overrides])
        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command in {'train-cuda', 'train-torch'}:
        backend = 'cuda_legacy' if args.command == 'train-cuda' else 'torch'
        cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *_common_train_overrides(args), *args.overrides])
        run_dir = train_unified_from_config(cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'train-autograd':
        from minicnn.training.train_autograd import train_autograd_from_config
        cfg = load_flex_config(args.config if args.config else None, [*_common_train_overrides(args), *args.overrides])
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
                cfg = load_flex_config(args.config if args.config else None, [*_common_train_overrides(args), *compare_overrides])
                run_dir = train_autograd_from_config(cfg)
            else:
                cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *_common_train_overrides(args), *compare_overrides])
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
        cfg = _load_unified_config_or_exit(args.config, args.overrides)
        errors = validate_cuda_legacy_compatibility(cfg) if cfg.get('engine', {}).get('backend') == 'cuda_legacy' else []
        print(json.dumps({'ok': not errors, 'errors': errors, 'backend': cfg.get('engine', {}).get('backend')}, indent=2))
        return 0 if not errors else 2

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
