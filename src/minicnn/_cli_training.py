from __future__ import annotations

import contextlib
import io
import json
import time
import sys
from pathlib import Path
from typing import Any

from minicnn._cli_config import (
    _ensure_cuda_legacy_prereqs_or_exit,
    _load_flex_config_or_exit,
    _load_unified_config_or_exit,
)
from minicnn._cli_errors import _ensure_torch_device_supported_or_exit, _run_user_operation_or_exit
from minicnn._cli_output import _print_json
from minicnn.user_errors import format_user_error


COMPARE_BACKENDS = {'torch', 'cuda_legacy', 'autograd'}


def _train_native_failure_category(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return 'missing_resource'
    message = str(exc).lower()
    if 'does not support' in message or 'unsupported' in message or 'only supports' in message:
        return 'unsupported_config'
    return 'invalid_input_or_environment'


def _exit_train_native_user_error(exc: Exception) -> None:
    category = _train_native_failure_category(exc)
    rendered = str(exc)
    if not rendered.startswith('[ERROR] '):
        rendered = format_user_error(
            'train-native failed because the provided config or environment is outside the validated support boundary.',
            cause=str(exc),
            fix='Review validate-cuda-native-config output, config values, dataset paths, and backend support limits, then retry.',
            example='minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml',
        )
    print(f'{rendered}\nCategory: {category}', file=sys.stderr)
    raise SystemExit(2)


def common_train_overrides(args) -> list[str]:
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


def _read_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    return summary if isinstance(summary, dict) else {}


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


def benchmark_fields(
    backend: str,
    cfg: dict[str, Any],
    run_dir: Path,
    elapsed_s: float,
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    variant = ''
    if isinstance(summary, dict):
        variant = str(summary.get('variant') or '')
    if not variant:
        if backend == 'cuda_legacy':
            variant = runtime_cfg.get('cuda_variant')
        elif backend == 'torch':
            variant = train_cfg.get('device')
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


def _compare_row(
    backend: str,
    cfg: dict[str, Any],
    run_dir: Path,
    elapsed_s: float,
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary if isinstance(summary, dict) else {}
    periodic_checkpoints = summary.get('periodic_checkpoints')
    if not isinstance(periodic_checkpoints, list):
        periodic_checkpoints = []
    row = {
        'backend': backend,
        'run_dir': str(run_dir),
        'schema_version': summary.get('schema_version'),
        'artifact_kind': summary.get('artifact_kind'),
        'status': summary.get('status'),
        'selected_backend': summary.get('selected_backend', backend),
        'effective_backend': summary.get('effective_backend', backend),
        'best_model_path': summary.get('best_model_path'),
        'periodic_checkpoints': periodic_checkpoints,
        'num_periodic_checkpoints': len(periodic_checkpoints),
        'test_loss': summary.get('test_loss'),
        'test_acc': summary.get('test_acc'),
    }
    row.update(benchmark_fields(backend, cfg, run_dir, elapsed_s, summary=summary))
    return row


def compare_backends_and_overrides(args) -> tuple[list[str], list[str]]:
    if args.backends:
        backends = []
        extra_overrides = []
        for token in args.backends:
            if token in COMPARE_BACKENDS and not extra_overrides:
                backends.append(token)
            elif '=' in token:
                extra_overrides.append(token)
            else:
                expected = ', '.join(sorted(COMPARE_BACKENDS))
                raise ValueError(f"invalid compare backend {token!r}; expected one of: {expected}")
        return backends, [*args.overrides, *extra_overrides]
    backends = [b for b in (args.backend_a, args.backend_b) if b] or ['torch', 'cuda_legacy']
    return backends, args.overrides


def _print_run_dir(run_dir: Path) -> int:
    print(f'Artifacts written to: {run_dir}')
    return 0


@contextlib.contextmanager
def _training_output_scope(args):
    if getattr(args, 'quiet', False) and not getattr(args, 'verbose', False):
        with contextlib.redirect_stdout(io.StringIO()):
            yield
        return
    yield


def handle_train_flex(args) -> int:
    from minicnn.flex.trainer import train_from_config

    cfg = _load_flex_config_or_exit(args.config, [*common_train_overrides(args), *args.overrides])
    _ensure_torch_device_supported_or_exit(cfg, 'train-flex')
    with _training_output_scope(args):
        run_dir = _run_user_operation_or_exit(train_from_config, cfg)
    return _print_run_dir(run_dir)


def _run_unified_training(cfg: dict[str, Any], *, torch_command_name: str) -> Path:
    backend = str(cfg.get('engine', {}).get('backend', 'torch'))
    if backend == 'torch':
        _ensure_torch_device_supported_or_exit(cfg, torch_command_name)
    elif backend == 'cuda_legacy':
        _ensure_cuda_legacy_prereqs_or_exit(cfg)
    from minicnn.unified.trainer import train_unified_from_config

    return _run_user_operation_or_exit(train_unified_from_config, cfg)


def handle_train_dual(args) -> int:
    cfg = _load_unified_config_or_exit(args.config, [*common_train_overrides(args), *args.overrides])
    with _training_output_scope(args):
        run_dir = _run_unified_training(cfg, torch_command_name='train-dual with engine.backend=torch')
    return _print_run_dir(run_dir)


def handle_train_dual_alias(args) -> int:
    backend = 'cuda_legacy' if args.command == 'train-cuda' else 'torch'
    cfg = _load_unified_config_or_exit(
        args.config,
        [f'engine.backend={backend}', *common_train_overrides(args), *args.overrides],
    )
    with _training_output_scope(args):
        run_dir = _run_unified_training(cfg, torch_command_name='train-dual with engine.backend=torch')
    return _print_run_dir(run_dir)


def handle_train_autograd(args) -> int:
    from minicnn.training.train_autograd import train_autograd_from_config

    cfg = _load_flex_config_or_exit(args.config if args.config else None, [*common_train_overrides(args), *args.overrides])
    with _training_output_scope(args):
        run_dir = _run_user_operation_or_exit(train_autograd_from_config, cfg)
    return _print_run_dir(run_dir)


def handle_compare(args, parser) -> int:
    rows = []
    try:
        backends, compare_overrides = compare_backends_and_overrides(args)
    except ValueError as exc:
        parser.error(str(exc))
    for backend in backends:
        t0 = time.perf_counter()
        if backend == 'autograd':
            from minicnn.training.train_autograd import train_autograd_from_config

            cfg = _load_flex_config_or_exit(args.config if args.config else None, [*common_train_overrides(args), *compare_overrides])
            with _training_output_scope(args):
                run_dir = _run_user_operation_or_exit(train_autograd_from_config, cfg)
        else:
            cfg = _load_unified_config_or_exit(args.config, [f'engine.backend={backend}', *common_train_overrides(args), *compare_overrides])
            if backend == 'torch':
                _ensure_torch_device_supported_or_exit(cfg, 'compare with engine.backend=torch')
            elif backend == 'cuda_legacy':
                _ensure_cuda_legacy_prereqs_or_exit(cfg)
            from minicnn.unified.trainer import train_unified_from_config
            with _training_output_scope(args):
                run_dir = _run_user_operation_or_exit(train_unified_from_config, cfg)
        elapsed = time.perf_counter() - t0
        summary = _read_summary(run_dir)
        rows.append(_compare_row(backend, cfg, run_dir, elapsed, summary=summary))
    _print_json({'runs': rows})
    return 0


def handle_train_native(args) -> int:
    from minicnn.cuda_native.contract import emit_experimental_warning
    from minicnn.cuda_native.api import (
        assess_cuda_native_support_tier,
        get_capability_summary as get_cuda_native_summary,
    )
    from minicnn.unified.trainer import train_unified_from_config

    cfg = _load_unified_config_or_exit(args.config, ['engine.backend=cuda_native', *common_train_overrides(args), *args.overrides])
    emit_experimental_warning(
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
        'support_tier_assessment': assess_cuda_native_support_tier(cfg),
    })
    try:
        with _training_output_scope(args):
            run_dir = train_unified_from_config(cfg)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        _exit_train_native_user_error(exc)
    return _print_run_dir(run_dir)
