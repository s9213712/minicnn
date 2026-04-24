from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.loss import bce_with_logits_loss, cross_entropy_loss, mse_loss
from minicnn.paths import BEST_MODELS_ROOT
from minicnn.schedulers.cosine import CosineAnnealingLR
from minicnn.schedulers.plateau import ReduceLROnPlateau
from minicnn.schedulers.step import StepLR

TRAINING_SUMMARY_SCHEMA_NAME = 'minicnn.cuda_native.training.summary'
TRAINING_SUMMARY_SCHEMA_VERSION = 1
TRAINING_METRICS_SCHEMA_NAME = 'minicnn.cuda_native.training.metrics.epoch'
TRAINING_METRICS_SCHEMA_VERSION = 1
CHECKPOINT_CONTRACT_VERSION = 1
EXECUTION_MODE_REFERENCE_NUMPY = 'reference_numpy'
EXECUTION_MODE_GPU_NATIVE = 'gpu_native'
EXECUTION_DEVICE_CPU = 'cpu'
EXECUTION_DEVICE_GPU = 'gpu'


def _sanitize_amp_runtime(amp_runtime: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(amp_runtime or {})
    sanitized: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            sanitized[key] = value
            continue
        if isinstance(value, np.generic):
            sanitized[key] = value.item()
            continue
        if key == 'params_fp16_cache' and isinstance(value, dict):
            sanitized['cached_param_tensors'] = int(len(value))
            continue
    return sanitized


def _sanitize_optimizer_runtime(optimizer_runtime: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(optimizer_runtime or {})
    sanitized: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            sanitized[key] = value
            continue
        if isinstance(value, np.generic):
            sanitized[key] = value.item()
    return sanitized


def _build_efficiency_summary(
    *,
    planner_summary: dict[str, Any],
    amp_runtime: dict[str, Any],
    optimizer_runtime: dict[str, Any],
) -> dict[str, Any]:
    steps = max(1, int(optimizer_runtime.get('steps', 0) or 0))
    state_allocations = int(optimizer_runtime.get('state_tensor_allocations', 0) or 0)
    state_updates = int(optimizer_runtime.get('state_tensor_updates', 0) or 0)
    scratch_allocations = int(optimizer_runtime.get('scratch_tensor_allocations', 0) or 0)
    scratch_updates = int(optimizer_runtime.get('scratch_tensor_updates', 0) or 0)
    grad_allocations = int(optimizer_runtime.get('grad_buffer_allocations', 0) or 0)
    grad_reuses = int(optimizer_runtime.get('grad_buffer_reuses', 0) or 0)
    grad_resets = int(optimizer_runtime.get('grad_buffer_reset_events', 0) or 0)
    grad_capacity_tensors = int(optimizer_runtime.get('grad_buffer_tensor_count', 0) or 0)
    grad_capacity_bytes = int(optimizer_runtime.get('grad_buffer_total_bytes', 0) or 0)
    grad_active_tensors = int(optimizer_runtime.get('grad_buffer_active_tensor_count', 0) or 0)
    grad_active_bytes = int(optimizer_runtime.get('grad_buffer_active_total_bytes', 0) or 0)
    cache_allocations = int(amp_runtime.get('cache_allocations', 0) or 0)
    cache_hits = int(amp_runtime.get('cache_hits', 0) or 0)
    cache_updates = int(amp_runtime.get('cache_updates', 0) or 0)
    cache_total = cache_allocations + cache_hits
    grad_total = grad_allocations + grad_reuses
    peak_live = int(planner_summary.get('peak_live_bytes', 0) or 0)
    total_bytes = int(planner_summary.get('total_bytes', 0) or 0)
    return {
        'state_allocations_per_step': round(state_allocations / float(steps), 6),
        'state_updates_per_step': round(state_updates / float(steps), 6),
        'scratch_allocations_per_step': round(scratch_allocations / float(steps), 6),
        'scratch_updates_per_step': round(scratch_updates / float(steps), 6),
        'grad_buffer_allocations_per_step': round(grad_allocations / float(steps), 6),
        'grad_buffer_resets_per_step': round(grad_resets / float(steps), 6),
        'grad_buffer_reuse_ratio': round(grad_reuses / float(grad_total), 6) if grad_total > 0 else 0.0,
        'grad_buffer_active_tensor_fraction': (
            round(grad_active_tensors / float(grad_capacity_tensors), 6) if grad_capacity_tensors > 0 else 0.0
        ),
        'grad_buffer_active_byte_fraction': (
            round(grad_active_bytes / float(grad_capacity_bytes), 6) if grad_capacity_bytes > 0 else 0.0
        ),
        'amp_cache_hit_ratio': round(cache_hits / float(cache_total), 6) if cache_total > 0 else 0.0,
        'amp_cache_updates_per_hit': round(cache_updates / float(max(cache_hits, 1)), 6) if cache_hits > 0 else 0.0,
        'planner_peak_live_fraction': round(peak_live / float(total_bytes), 6) if total_bytes > 0 else 0.0,
    }


def _build_bottleneck_summary(
    *,
    efficiency_summary: dict[str, Any],
    runtime_profile: dict[str, Any],
) -> dict[str, Any]:
    hotspot_diff = dict(runtime_profile.get('hotspot_diff', {}) or {})
    bottleneck_hints: list[dict[str, Any]] = []
    planner_peak_live_fraction = float(efficiency_summary.get('planner_peak_live_fraction', 0.0) or 0.0)
    amp_cache_hit_ratio = float(efficiency_summary.get('amp_cache_hit_ratio', 0.0) or 0.0)
    grad_buffer_active_byte_fraction = float(
        efficiency_summary.get('grad_buffer_active_byte_fraction', 0.0) or 0.0
    )
    state_allocations_per_step = float(efficiency_summary.get('state_allocations_per_step', 0.0) or 0.0)
    train_eval_delta_ms = float(hotspot_diff.get('delta_ms', 0.0) or 0.0)

    if planner_peak_live_fraction >= 0.85:
        bottleneck_hints.append(
            {
                'type': 'planner_memory_pressure',
                'severity': 'high',
                'value': round(planner_peak_live_fraction, 6),
            }
        )
    if amp_cache_hit_ratio < 0.75:
        bottleneck_hints.append(
            {
                'type': 'amp_cache_churn',
                'severity': 'medium',
                'value': round(amp_cache_hit_ratio, 6),
            }
        )
    if grad_buffer_active_byte_fraction < 0.5:
        bottleneck_hints.append(
            {
                'type': 'grad_buffer_over_retained',
                'severity': 'medium',
                'value': round(grad_buffer_active_byte_fraction, 6),
            }
        )
    if state_allocations_per_step > 0.25:
        bottleneck_hints.append(
            {
                'type': 'optimizer_state_churn',
                'severity': 'medium',
                'value': round(state_allocations_per_step, 6),
            }
        )
    if abs(train_eval_delta_ms) > 0.1:
        bottleneck_hints.append(
            {
                'type': 'train_eval_hotspot_skew',
                'severity': 'info',
                'value': round(train_eval_delta_ms, 3),
            }
        )

    dominant_hint = bottleneck_hints[0] if bottleneck_hints else None
    return {
        'dominant': dominant_hint,
        'hints': bottleneck_hints,
        'hotspot_bottleneck': hotspot_diff.get('bottleneck_summary', {}),
    }


def load_numpy_data(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    dtype = str(dataset_cfg.get('type', 'random'))

    if dtype == 'random':
        from minicnn.flex._datasets import DATASET_ARRAY_LOADERS

        return DATASET_ARRAY_LOADERS['random'](dataset_cfg, train_cfg)

    if dtype == 'cifar10':
        from pathlib import Path as P

        from minicnn.config.parsing import parse_bool
        from minicnn.data.cifar10 import load_cifar10, normalize_cifar

        data_root = dataset_cfg.get('data_root', 'data/cifar-10-batches-py')
        n_train = int(dataset_cfg.get('num_samples', 512))
        n_val = int(dataset_cfg.get('val_samples', 128))
        seed = int(dataset_cfg.get('seed', 42))
        try:
            x_tr, y_tr, x_v, y_v, _, _ = load_cifar10(
                data_root=P(data_root),
                n_train=n_train,
                n_val=n_val,
                seed=seed,
                download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f'cuda_native: CIFAR-10 data not found at {data_root!r}. '
                'Set dataset.download=true or use dataset.type=random.'
            ) from exc
        return normalize_cifar(x_tr), y_tr, normalize_cifar(x_v), y_v

    if dtype == 'mnist':
        from pathlib import Path as P

        from minicnn.config.parsing import parse_bool
        from minicnn.data.mnist import load_mnist, normalize_mnist

        data_root = dataset_cfg.get('data_root', 'data/mnist')
        n_train = int(dataset_cfg.get('num_samples', 60000))
        n_val = int(dataset_cfg.get('val_samples', 10000))
        seed = int(dataset_cfg.get('seed', 42))
        try:
            x_tr, y_tr, x_v, y_v, _, _ = load_mnist(
                data_root=P(data_root),
                n_train=n_train,
                n_val=n_val,
                seed=seed,
                download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f'cuda_native: MNIST data not found at {data_root!r}. '
                'Set dataset.download=true or use dataset.type=random.'
            ) from exc
        return normalize_mnist(x_tr), y_tr, normalize_mnist(x_v), y_v

    raise ValueError(
        f"cuda_native does not support dataset.type={dtype!r}. Supported: random, cifar10, mnist."
    )


def evaluate_native_graph(
    graph: NativeGraph,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    batch_size: int,
    loss_type: str,
    *,
    amp_enabled: bool = False,
    device_runtime: Any | None = None,
) -> dict[str, float]:
    fwd = ForwardExecutor()
    out_name = graph.output_spec.name
    total_loss = 0.0
    correct = 0
    seen = 0
    n = x.shape[0]
    eval_params = params
    if amp_enabled:
        eval_params = {
            key: (
                value.astype(np.float16)
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating)
                else value
            )
            for key, value in params.items()
        }

    def _run_forward_with_device_runtime(x_batch: np.ndarray) -> np.ndarray:
        assert device_runtime is not None
        input_name = graph.input_spec.name
        staged_input = device_runtime.stage_to_device(x_batch, name=input_name, prefer_reserved=True)
        ctx = fwd.run(graph, {input_name: staged_input.data}, params=eval_params)
        logits = ctx[out_name]
        staged_output = device_runtime.allocate_staging_buffer(logits.shape, dtype=logits.dtype, name=out_name)
        np.copyto(staged_output.data, logits)
        host_logits = device_runtime.stage_to_host(staged_output, copy=True)
        device_runtime.release_buffer(staged_input)
        device_runtime.release_buffer(staged_output)
        device_runtime.record_execution(
            'eval_forward',
            input_name=input_name,
            output_name=out_name,
            node_count=len(graph.nodes),
        )
        device_runtime.synchronize('eval-forward')
        return host_logits

    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        fwd_input = xb.astype(np.float16) if amp_enabled else xb
        if device_runtime is not None:
            logits = _run_forward_with_device_runtime(fwd_input)
        else:
            ctx = fwd.run(graph, {graph.input_spec.name: fwd_input}, params=eval_params)
            logits = ctx[out_name]
        if loss_type == 'cross_entropy':
            loss_val, _ = cross_entropy_loss(logits, yb)
            preds = logits.argmax(axis=1)
            correct += int((preds == yb).sum())
        elif loss_type == 'bce_with_logits':
            loss_val, _ = bce_with_logits_loss(logits, yb.astype(np.float32))
            probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            preds = (probs >= 0.5).astype(np.int64)
            correct += int((preds == yb.reshape(-1).astype(np.int64)).sum())
        else:
            loss_val, _ = mse_loss(logits, yb.astype(np.float32))
            preds = logits.argmax(axis=1)
            correct += int((preds == yb).sum())
        total_loss += loss_val * xb.shape[0]
        seen += xb.shape[0]
    return {
        'loss': total_loss / max(seen, 1),
        'acc': correct / max(seen, 1),
    }


def make_scheduler(
    scheduler_cfg: dict[str, Any],
    optimizer_view: SimpleNamespace,
):
    if not bool(scheduler_cfg.get('enabled', False)):
        return None, None
    raw_type = str(scheduler_cfg.get('type', 'none') or 'none').lower()
    if raw_type in {'step', 'steplr'}:
        scheduler = StepLR(
            optimizer_view,
            step_size=int(scheduler_cfg.get('step_size', 10)),
            gamma=float(scheduler_cfg.get('gamma', 0.5)),
            min_lr=float(scheduler_cfg.get('min_lr', 0.0)),
        )
        return scheduler, 'step'
    if raw_type in {'cosine', 'cosineannealinglr'}:
        scheduler = CosineAnnealingLR(
            optimizer_view,
            T_max=int(scheduler_cfg.get('T_max', scheduler_cfg.get('t_max', 10))),
            lr_min=float(scheduler_cfg.get('lr_min', 0.0)),
        )
        return scheduler, 'cosine'
    if raw_type in {'plateau', 'reducelronplateau'}:
        scheduler = ReduceLROnPlateau(
            optimizer_view,
            factor=float(scheduler_cfg.get('factor', 0.5)),
            patience=int(scheduler_cfg.get('patience', 3)),
            min_lr=float(scheduler_cfg.get('min_lr', 1e-5)),
        )
        return scheduler, 'plateau'
    return None, None


def resolve_loss_type(loss_cfg: dict[str, Any]) -> str:
    loss_map = {
        'CrossEntropyLoss': 'cross_entropy',
        'BCEWithLogitsLoss': 'bce_with_logits',
        'MSELoss': 'mse',
    }
    loss_type_str = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    loss_type = loss_map.get(loss_type_str)
    if loss_type is None:
        raise ValueError(
            f'cuda_native does not support loss.type={loss_type_str!r}. '
            'Supported: CrossEntropyLoss, BCEWithLogitsLoss, MSELoss.'
        )
    return loss_type


def build_epoch_row(
    *,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, float],
    lr: float,
    epoch_time_s: float,
    amp_state: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    planner_state: dict[str, Any] | None = None,
    device_runtime_state: dict[str, Any] | None = None,
    support_tier_assessment: dict[str, Any] | None = None,
    execution_mode: str = EXECUTION_MODE_REFERENCE_NUMPY,
    tensor_execution_device: str = EXECUTION_DEVICE_CPU,
) -> dict[str, Any]:
    row = {
        'schema_name': TRAINING_METRICS_SCHEMA_NAME,
        'schema_version': TRAINING_METRICS_SCHEMA_VERSION,
        'artifact_kind': 'training_metrics_epoch',
        'execution_mode': execution_mode,
        'effective_execution_mode': execution_mode,
        'tensor_execution_device': tensor_execution_device,
        'tensors_ran_on': tensor_execution_device,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_acc': val_metrics['acc'],
        'lr': lr,
        'epoch_time_s': epoch_time_s,
    }
    if amp_state:
        row['amp'] = dict(amp_state)
    if optimizer_state:
        row['optimizer_runtime'] = _sanitize_optimizer_runtime(optimizer_state)
    if planner_state:
        row['planner'] = dict(planner_state)
    if device_runtime_state:
        row['device_runtime'] = dict(device_runtime_state)
    if support_tier_assessment:
        row['support_tier_assessment'] = dict(support_tier_assessment)
    if amp_state or optimizer_state or planner_state or device_runtime_state:
        row['efficiency'] = _build_efficiency_summary(
            planner_summary=dict(planner_state or {}),
            amp_runtime=dict(amp_state or {}),
            optimizer_runtime=_sanitize_optimizer_runtime(optimizer_state),
        )
    return row


def epoch_log_message(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    val_acc: float,
    lr: float,
    epoch_time_s: float,
    amp_state: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> str:
    message = (
        f"[cuda_native] Epoch {epoch}/{epochs}: "
        f"train_loss={train_loss:.4f}, "
        f"val_acc={val_acc * 100:.2f}%, "
        f"lr={lr:.6f}, "
        f"time={epoch_time_s:.1f}s"
    )
    if amp_state:
        parts: list[str] = []
        if 'loss_scale' in amp_state:
            parts.append(f"amp_scale={float(amp_state['loss_scale']):.1f}")
        if 'skipped_steps_epoch' in amp_state or 'overflow_steps_epoch' in amp_state:
            parts.append(
                f"amp_skip={int(amp_state.get('skipped_steps_epoch', 0))}"
                f"/ovf={int(amp_state.get('overflow_steps_epoch', 0))}"
            )
        if 'cache_hits_epoch' in amp_state or 'cache_allocations_epoch' in amp_state:
            parts.append(
                f"amp_cache=h{int(amp_state.get('cache_hits_epoch', 0))}"
                f"/u{int(amp_state.get('cache_updates_epoch', 0))}"
                f"/a{int(amp_state.get('cache_allocations_epoch', 0))}"
            )
        if parts:
            message += ", " + ", ".join(parts)
    if optimizer_state:
        message += (
            f", opt_state=t{int(optimizer_state.get('state_tensor_count', 0))}"
            f"/{float(optimizer_state.get('state_total_kb', 0.0)):.1f}kb"
            f",a{int(optimizer_state.get('state_tensor_allocations_epoch', 0))}"
            f"/u{int(optimizer_state.get('state_tensor_updates_epoch', 0))}"
        )
    return message


def best_checkpoint_path(run_dir: Path) -> Path:
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return BEST_MODELS_ROOT / f'{run_dir.name}_best.npz'


def build_training_summary(
    *,
    run_dir: Path,
    best_path: Path,
    best_val_acc: float,
    input_shape: tuple[int, ...],
    model_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    optimizer_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    amp_runtime: dict[str, Any] | None,
    optimizer_runtime: dict[str, Any] | None,
    planner_summary: dict[str, Any] | None,
    runtime_profile: dict[str, Any] | None,
    epochs: int,
    capabilities: dict[str, Any],
    support_tier_assessment: dict[str, Any] | None = None,
    execution_mode: str = EXECUTION_MODE_REFERENCE_NUMPY,
    tensor_execution_device: str = EXECUTION_DEVICE_CPU,
    device_runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optim_type = str(optimizer_cfg.get('type', 'SGD'))
    optimizer_summary = {
        'type': optim_type,
        'lr': float(optimizer_cfg.get('lr', 0.01)),
        'weight_decay': float(optimizer_cfg.get('weight_decay', 0.0)),
        'grad_clip_global': float(optimizer_cfg.get('grad_clip_global', 0.0)),
    }
    if optim_type in {'SGD', 'RMSprop'}:
        optimizer_summary['momentum'] = float(optimizer_cfg.get('momentum', 0.0))
    if optim_type in {'Adam', 'AdamW'}:
        optimizer_summary['beta1'] = float(optimizer_cfg.get('beta1', 0.9))
        optimizer_summary['beta2'] = float(optimizer_cfg.get('beta2', 0.999))
        optimizer_summary['eps'] = float(optimizer_cfg.get('eps', 1e-8))
    if optim_type == 'RMSprop':
        optimizer_summary['alpha'] = float(optimizer_cfg.get('alpha', 0.99))
    sanitized_amp_runtime = _sanitize_amp_runtime(amp_runtime)
    sanitized_optimizer_runtime = _sanitize_optimizer_runtime(optimizer_runtime)
    planner_payload = dict(planner_summary or {})
    efficiency_summary = _build_efficiency_summary(
        planner_summary=planner_payload,
        amp_runtime=sanitized_amp_runtime,
        optimizer_runtime=sanitized_optimizer_runtime,
    )
    performance_report = {
        'planner': planner_payload,
        'amp': sanitized_amp_runtime,
        'optimizer': sanitized_optimizer_runtime,
        'efficiency': efficiency_summary,
        'bottlenecks': _build_bottleneck_summary(
            efficiency_summary=efficiency_summary,
            runtime_profile=dict(runtime_profile or {}),
        ),
        'runtime': {
            **dict(runtime_profile or {}),
            'execution_mode': execution_mode,
            'effective_execution_mode': execution_mode,
            'tensor_execution_device': tensor_execution_device,
            'tensors_ran_on': tensor_execution_device,
            'device_runtime': dict(device_runtime_state or {}),
        },
        'training': {
            'batch_size': int(train_cfg.get('batch_size', 64)),
            'grad_accum_steps': int(train_cfg.get('grad_accum_steps', 1)),
            'amp_enabled': bool(train_cfg.get('amp', False)),
            'support_tier': dict(support_tier_assessment or {}),
        },
    }
    return {
        'schema_name': TRAINING_SUMMARY_SCHEMA_NAME,
        'schema_version': TRAINING_SUMMARY_SCHEMA_VERSION,
        'artifact_kind': 'training_run_summary',
        'status': 'ok',
        'execution_mode': execution_mode,
        'selected_execution_mode': execution_mode,
        'effective_execution_mode': execution_mode,
        'tensor_execution_device': tensor_execution_device,
        'tensors_ran_on': tensor_execution_device,
        'device_runtime': dict(device_runtime_state or {}),
        'selected_backend': 'cuda_native',
        'effective_backend': 'cuda_native',
        'variant': 'reference',
        'run_dir': str(run_dir),
        'best_model_path': str(best_path),
        'best_val_acc': best_val_acc,
        'input_shape': list(input_shape),
        'model_layers': [layer.get('type') for layer in model_cfg.get('layers', [])],
        'optimizer': optimizer_summary,
        'scheduler': scheduler_cfg.get('type') if bool(scheduler_cfg.get('enabled', False)) else None,
        'loss': {
            'type': loss_cfg.get('type', 'CrossEntropyLoss'),
            'label_smoothing': float(loss_cfg.get('label_smoothing', 0.0)),
        },
        'grad_accum_steps': int(train_cfg.get('grad_accum_steps', 1)),
        'amp': bool(train_cfg.get('amp', False)),
        'amp_config': {
            'enabled': bool(train_cfg.get('amp', False)),
            'loss_scale': float(train_cfg.get('amp_loss_scale', 128.0)),
            'dynamic_scale': bool(train_cfg.get('amp_dynamic_scale', True)),
            'scale_growth': float(train_cfg.get('amp_scale_growth', 2.0)),
            'scale_backoff': float(train_cfg.get('amp_scale_backoff', 0.5)),
            'scale_window': int(train_cfg.get('amp_scale_window', 200)),
        },
        'amp_runtime': sanitized_amp_runtime,
        'optimizer_runtime': sanitized_optimizer_runtime,
        'checkpoint_contract': {
            'format': 'npz',
            'version': CHECKPOINT_CONTRACT_VERSION,
            'best_model_path_key': 'best_model_path',
        },
        'planner': planner_payload,
        'performance_report': performance_report,
        'support_tier_assessment': dict(support_tier_assessment or {}),
        'epochs': epochs,
        'periodic_checkpoints': [],
        'test_loss': None,
        'test_acc': None,
        'capabilities': capabilities,
    }
