from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from minicnn._cli_config import (
    _ensure_cuda_legacy_prereqs_or_exit,
    _load_flex_config_or_exit,
    _load_unified_config_or_exit,
    _resolve_cli_config_path,
)
from minicnn._cli_errors import (
    _ensure_torch_device_supported_or_exit,
    _run_user_operation_or_exit,
)
from minicnn._cli_output import (
    _add_format_arg,
    _print_json,
)
from minicnn._cli_readonly import (
    handle_compile,
    handle_config_template,
    handle_cuda_native_capabilities,
    handle_doctor,
    handle_dual_config_template,
    handle_export_torch_checkpoint,
    handle_healthcheck,
    handle_info,
    handle_inspect_checkpoint,
    handle_list_dual_components,
    handle_list_flex_components,
    handle_show_cuda_mapping,
    handle_show_graph,
    handle_show_model,
    handle_smoke,
    handle_validate_config,
    handle_validate_cuda_native_config,
    handle_validate_dual_config,
)
from minicnn.core.build import build_native, check_native


_COMPARE_BACKENDS = {'torch', 'cuda_legacy', 'autograd'}


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
    p_info = sub.add_parser('info', help='Show important project paths and summary')
    _add_format_arg(p_info)
    p_doctor = sub.add_parser('doctor', help='Run first-run diagnostics for paths, data, and native CUDA')
    _add_format_arg(p_doctor)
    p_health = sub.add_parser('healthcheck', help='Validate framework wiring and native artifacts')
    _add_format_arg(p_health)
    p_smoke = sub.add_parser('smoke', help='Run a compact first-run self-check across configs, compiler, and backend validation')
    _add_format_arg(p_smoke)
    sub.add_parser('list-flex-components', help='List configurable built-in components')
    sub.add_parser('list-dual-components', help='List dual-backend components and cuda_legacy subset')
    sub.add_parser('config-template', help='Print the PyTorch-flex config template')
    sub.add_parser('dual-config-template', help='Print the dual-backend unified config template')
    p_inspect_ckpt = sub.add_parser('inspect-checkpoint', help='Inspect a saved model/checkpoint artifact (.pt/.pth/.npz)')
    p_inspect_ckpt.add_argument('--path', required=True, type=str)
    _add_format_arg(p_inspect_ckpt)
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
    _add_format_arg(p_validate)

    p_map = sub.add_parser('show-cuda-mapping', help='Show how a unified config maps onto the handcrafted CUDA backend')
    p_map.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_map.add_argument('overrides', nargs='*')
    _add_format_arg(p_map)

    p_validate_any = sub.add_parser('validate-config', help='Validate shared config shape and backend compatibility')
    p_validate_any.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate_any.add_argument('overrides', nargs='*')
    _add_format_arg(p_validate_any)

    p_compile = sub.add_parser('compile', help='Trace and optimize a model config into MiniCNN IR')
    p_compile.add_argument('--config', type=str, default='configs/autograd_tiny.yaml')
    p_compile.add_argument('overrides', nargs='*')
    p_show_model = sub.add_parser('show-model', help='Render a human-oriented model summary from config/frontend structure')
    p_show_model.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    p_show_model.add_argument('overrides', nargs='*')
    _add_format_arg(p_show_model)
    p_show_graph = sub.add_parser('show-graph', help='Render the canonical primitive graph traced from config')
    p_show_graph.add_argument('--config', type=str, default='configs/flex_cnn.yaml')
    p_show_graph.add_argument('overrides', nargs='*')
    _add_format_arg(p_show_graph)

    p_train_native = sub.add_parser('train-native', help='[EXPERIMENTAL] Train with cuda_native backend (research prototype)')
    p_train_native.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_train_native.add_argument('overrides', nargs='*')

    p_validate_native = sub.add_parser('validate-cuda-native-config', help='[EXPERIMENTAL] Validate config against cuda_native constraints')
    p_validate_native.add_argument('--config', type=str, default='configs/dual_backend_cnn.yaml')
    p_validate_native.add_argument('overrides', nargs='*')
    _add_format_arg(p_validate_native)

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
        return handle_info(args)

    if args.command == 'healthcheck':
        return handle_healthcheck(args)

    if args.command == 'smoke':
        return handle_smoke(args)

    if args.command == 'doctor':
        return handle_doctor(args)

    if args.command == 'list-flex-components':
        return handle_list_flex_components()

    if args.command == 'list-dual-components':
        return handle_list_dual_components()

    if args.command == 'config-template':
        return handle_config_template()

    if args.command == 'dual-config-template':
        return handle_dual_config_template()

    if args.command == 'inspect-checkpoint':
        return handle_inspect_checkpoint(args)

    if args.command == 'export-torch-checkpoint':
        return handle_export_torch_checkpoint(args)

    if args.command == 'train-flex':
        from minicnn.flex.trainer import train_from_config

        cfg = _load_flex_config_or_exit(args.config, [*_common_train_overrides(args), *args.overrides])
        _ensure_torch_device_supported_or_exit(cfg, 'train-flex')
        run_dir = _run_user_operation_or_exit(train_from_config, cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'validate-dual-config':
        return handle_validate_dual_config(args)

    if args.command == 'show-cuda-mapping':
        return handle_show_cuda_mapping(args)

    if args.command in {'train', 'train-dual'}:
        cfg = _load_unified_config_or_exit(args.config, [*_common_train_overrides(args), *args.overrides])
        backend = str(cfg.get('engine', {}).get('backend', 'torch'))
        if backend == 'torch':
            _ensure_torch_device_supported_or_exit(cfg, 'train-dual with engine.backend=torch')
        elif backend == 'cuda_legacy':
            _ensure_cuda_legacy_prereqs_or_exit(cfg)
        from minicnn.unified.trainer import train_unified_from_config

        run_dir = _run_user_operation_or_exit(train_unified_from_config, cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command in {'train-cuda', 'train-torch'}:
        backend = 'cuda_legacy' if args.command == 'train-cuda' else 'torch'
        cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *_common_train_overrides(args), *args.overrides])
        if backend == 'torch':
            _ensure_torch_device_supported_or_exit(cfg, 'train-dual with engine.backend=torch')
        else:
            _ensure_cuda_legacy_prereqs_or_exit(cfg)
        from minicnn.unified.trainer import train_unified_from_config

        run_dir = _run_user_operation_or_exit(train_unified_from_config, cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'train-autograd':
        from minicnn.training.train_autograd import train_autograd_from_config
        cfg = _load_flex_config_or_exit(args.config if args.config else None, [*_common_train_overrides(args), *args.overrides])
        run_dir = _run_user_operation_or_exit(train_autograd_from_config, cfg)
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
                run_dir = _run_user_operation_or_exit(train_autograd_from_config, cfg)
            else:
                cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *_common_train_overrides(args), *compare_overrides])
                if backend == 'torch':
                    _ensure_torch_device_supported_or_exit(cfg, 'compare with engine.backend=torch')
                elif backend == 'cuda_legacy':
                    _ensure_cuda_legacy_prereqs_or_exit(cfg)
                from minicnn.unified.trainer import train_unified_from_config
                run_dir = _run_user_operation_or_exit(train_unified_from_config, cfg)
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
        _print_json({'runs': rows})
        return 0

    if args.command == 'validate-config':
        return handle_validate_config(args)

    if args.command == 'validate-cuda-native-config':
        return handle_validate_cuda_native_config(args)

    if args.command == 'cuda-native-capabilities':
        return handle_cuda_native_capabilities()

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
        _print_json({
            'backend': 'cuda_native',
            'status': 'experimental',
            'validated_support_boundary': {
                'datasets': summary.get('supported_datasets', []),
                'losses': summary.get('supported_losses', []),
                'optimizers': summary.get('supported_optimizers', []),
                'schedulers': summary.get('supported_schedulers', []),
                'ops': summary.get('supported_ops', []),
            },
        })
        run_dir = _run_user_operation_or_exit(train_unified_from_config, cfg)
        print(f'Artifacts written to: {run_dir}')
        return 0

    if args.command == 'compile':
        return handle_compile(args)

    if args.command == 'show-model':
        return handle_show_model(args)

    if args.command == 'show-graph':
        return handle_show_graph(args)

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
