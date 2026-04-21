"""Bridge: shared unified config → cuda_native backend.

Responsibilities:
- Validate the config against cuda_native constraints (clear errors, no silent fallback)
- Translate config layers into a NativeGraph
- Initialize parameters
- Load dataset as numpy arrays (no torch dependency)
- Run minimal training loop and return a summary dict
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.api import (
    build_cuda_native_graph,
    get_capability_summary,
    validate_cuda_native_config,
)
from minicnn.cuda_native.backward import BackwardExecutor
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.loss import cross_entropy_loss, mse_loss
from minicnn.cuda_native.training import sgd_update, train_step
from minicnn.flex.runtime import create_run_dir, dump_summary
from minicnn.schedulers.cosine import CosineAnnealingLR
from minicnn.schedulers.plateau import ReduceLROnPlateau
from minicnn.schedulers.step import StepLR


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def check_config(cfg: dict[str, Any]) -> list[str]:
    """Return validation errors for *cfg* against cuda_native constraints."""
    return validate_cuda_native_config(cfg)


def get_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_capability_summary()


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def _init_params(graph: NativeGraph, seed: int = 42) -> dict[str, np.ndarray]:
    """Initialise trainable parameters for every node in *graph*.

    Uses He-normal for Conv2d/Linear weights and zeros for biases.
    Returns a flat dict with keys '_w_{node.name}' and '_b_{node.name}'.
    """
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}

    for node in graph.nodes:
        if node.op_type == 'Conv2d':
            s = node.input_specs[0].shape if node.input_specs else None
            out_spec = node.output_specs[0] if node.output_specs else None
            if s is None or out_spec is None:
                continue
            c_in = s[1]
            c_out = out_spec.shape[1]
            kh = kw = node.attrs.get('kernel_size', 3)
            fan_in = int(c_in * kh * kw)
            w = rng.standard_normal((c_out, c_in, kh, kw)) * np.sqrt(2.0 / fan_in)
            params[f'_w_{node.name}'] = w.astype(np.float32)

        elif node.op_type == 'Linear':
            s = node.input_specs[0].shape if node.input_specs else None
            out_spec = node.output_specs[0] if node.output_specs else None
            if s is None or out_spec is None:
                continue
            in_f = s[1]
            out_f = out_spec.shape[1]
            w = rng.standard_normal((out_f, in_f)) * np.sqrt(2.0 / in_f)
            params[f'_w_{node.name}'] = w.astype(np.float32)
            params[f'_b_{node.name}'] = np.zeros(out_f, dtype=np.float32)
        elif node.op_type == 'BatchNorm2d':
            s = node.input_specs[0].shape if node.input_specs else None
            if s is None:
                continue
            channels = int(s[1])
            params[f'_w_{node.name}'] = np.ones(channels, dtype=np.float32)
            params[f'_b_{node.name}'] = np.zeros(channels, dtype=np.float32)
            params[f'_running_mean_{node.name}'] = np.zeros(channels, dtype=np.float32)
            params[f'_running_var_{node.name}'] = np.ones(channels, dtype=np.float32)

    return params


# ---------------------------------------------------------------------------
# Dataset loading (pure numpy, no torch dependency)
# ---------------------------------------------------------------------------

def _load_numpy_data(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val data as numpy arrays from the dataset config.

    Supports:
        dataset.type = random   — synthetic random data, always available
        dataset.type = cifar10  — requires data files on disk
        dataset.type = mnist    — requires data files on disk

    Raises:
        ValueError for unsupported or unavailable datasets with clear message.
    """
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    dtype = str(dataset_cfg.get('type', 'random'))

    if dtype == 'random':
        from minicnn.flex.data import _random_dataset  # pure numpy, no torch
        return _random_dataset(dataset_cfg, train_cfg)

    if dtype == 'cifar10':
        from pathlib import Path as P
        from minicnn.data.cifar10 import load_cifar10, normalize_cifar
        from minicnn.config.parsing import parse_bool
        data_root = dataset_cfg.get('data_root', 'data/cifar-10-batches-py')
        n_train = int(dataset_cfg.get('num_samples', 512))
        n_val = int(dataset_cfg.get('val_samples', 128))
        seed = int(dataset_cfg.get('seed', 42))
        try:
            x_tr, y_tr, x_v, y_v, _, _ = load_cifar10(
                data_root=P(data_root), n_train=n_train, n_val=n_val, seed=seed,
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
        from minicnn.data.mnist import load_mnist, normalize_mnist
        from minicnn.config.parsing import parse_bool
        data_root = dataset_cfg.get('data_root', 'data/mnist')
        n_train = int(dataset_cfg.get('num_samples', 60000))
        n_val = int(dataset_cfg.get('val_samples', 10000))
        seed = int(dataset_cfg.get('seed', 42))
        try:
            x_tr, y_tr, x_v, y_v, _, _ = load_mnist(
                data_root=P(data_root), n_train=n_train, n_val=n_val, seed=seed,
                download=parse_bool(dataset_cfg.get('download', False), label='dataset.download'),
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f'cuda_native: MNIST data not found at {data_root!r}. '
                'Set dataset.download=true or use dataset.type=random.'
            ) from exc
        return normalize_mnist(x_tr), y_tr, normalize_mnist(x_v), y_v

    raise ValueError(
        f'cuda_native does not support dataset.type={dtype!r}. '
        'Supported: random, cifar10, mnist.'
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    graph: NativeGraph,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    batch_size: int,
    loss_type: str,
) -> dict[str, float]:
    fwd = ForwardExecutor()
    out_name = graph.output_spec.name
    total_loss = 0.0
    correct = 0
    seen = 0
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        ctx = fwd.run(graph, {graph.input_spec.name: xb}, params=params)
        logits = ctx[out_name]
        if loss_type == 'cross_entropy':
            loss_val, _ = cross_entropy_loss(logits, yb)
            preds = logits.argmax(axis=1)
            correct += int((preds == yb).sum())
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


def _make_scheduler(
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_cuda_native_training(cfg: dict[str, Any]) -> Path:
    """Run a full training loop for cuda_native and return the run directory.

    This is the backend-specific training entry point wired into
    unified/trainer.py when engine.backend = cuda_native.
    """
    errors = check_config(cfg)
    if errors:
        raise ValueError(
            'Config is not compatible with cuda_native:\n- ' + '\n- '.join(errors)
        )

    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    optim_cfg = cfg.get('optimizer', {})
    loss_cfg = cfg.get('loss', {})
    scheduler_cfg = cfg.get('scheduler', {})

    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    batch_size = int(train_cfg.get('batch_size', 64))
    epochs = int(train_cfg.get('epochs', 1))
    lr = float(optim_cfg.get('lr', 0.01))
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    momentum = float(optim_cfg.get('momentum', 0.0))
    grad_clip_global = float(optim_cfg.get('grad_clip_global', 0.0))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('seed', 42)))

    _loss_map = {'CrossEntropyLoss': 'cross_entropy', 'MSELoss': 'mse'}
    loss_type_str = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    loss_type = _loss_map.get(loss_type_str)
    if loss_type is None:
        raise ValueError(
            f'cuda_native does not support loss.type={loss_type_str!r}. '
            'Supported: CrossEntropyLoss, MSELoss.'
        )

    x_train, y_train, x_val, y_val = _load_numpy_data(cfg)
    graph = build_cuda_native_graph(model_cfg, (batch_size, *input_shape))
    params = _init_params(graph, seed=init_seed)
    run_dir = create_run_dir(cfg)
    metrics_path = run_dir / 'metrics.jsonl'

    fwd = ForwardExecutor()
    bwd = BackwardExecutor()
    optimizer_view = SimpleNamespace(lr=lr)
    scheduler, scheduler_kind = _make_scheduler(scheduler_cfg, optimizer_view)
    optimizer_state: dict[str, Any] = {}
    best_val_acc = float('-inf')
    best_params = dict(params)
    rng = np.random.default_rng(init_seed)

    with metrics_path.open('w', encoding='utf-8') as mf:
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            idx = rng.permutation(x_train.shape[0])
            x_shuf, y_shuf = x_train[idx], y_train[idx]
            running_loss = 0.0
            seen = 0
            for i in range(0, x_shuf.shape[0], batch_size):
                xb = x_shuf[i:i + batch_size]
                yb = y_shuf[i:i + batch_size]
                if xb.shape[0] == 0:
                    continue
                loss_val, params = train_step(
                    graph, xb, yb, params, lr=optimizer_view.lr,
                    loss_type=loss_type, weight_decay=weight_decay,
                    momentum=momentum,
                    optimizer_state=optimizer_state,
                    grad_clip_global=grad_clip_global,
                    fwd_executor=fwd, bwd_executor=bwd,
                )
                running_loss += loss_val * xb.shape[0]
                seen += xb.shape[0]
            train_loss = running_loss / max(seen, 1)
            val_metrics = _evaluate(graph, x_val, y_val, params, batch_size, loss_type)
            if scheduler is not None:
                if scheduler_kind == 'plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            epoch_time = time.perf_counter() - t0
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['acc'],
                'lr': float(optimizer_view.lr),
                'epoch_time_s': epoch_time,
            }
            mf.write(json.dumps(row) + '\n')
            mf.flush()
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_params = {k: v.copy() for k, v in params.items()}
            print(
                f"[cuda_native] Epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_acc={val_metrics['acc'] * 100:.2f}%, "
                f"lr={optimizer_view.lr:.6f}, "
                f"time={epoch_time:.1f}s"
            )

    # Save best weights
    from minicnn.paths import BEST_MODELS_ROOT
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    best_path = BEST_MODELS_ROOT / f'{run_dir.name}_best.npz'
    np.savez(str(best_path), **best_params)

    summary = {
        'selected_backend': 'cuda_native',
        'effective_backend': 'cuda_native',
        'run_dir': str(run_dir),
        'best_model_path': str(best_path),
        'best_val_acc': best_val_acc,
        'input_shape': list(input_shape),
        'model_layers': [layer.get('type') for layer in model_cfg.get('layers', [])],
        'optimizer': {
            'type': 'SGD',
            'lr': float(lr),
            'momentum': float(momentum),
            'grad_clip_global': float(grad_clip_global),
        },
        'scheduler': scheduler_cfg.get('type') if bool(scheduler_cfg.get('enabled', False)) else None,
        'loss': loss_cfg.get('type', 'CrossEntropyLoss'),
        'epochs': epochs,
        'capabilities': get_capability_summary(),
    }
    dump_summary(run_dir, summary)
    return run_dir
