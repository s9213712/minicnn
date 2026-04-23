from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.loss import cross_entropy_loss, mse_loss
from minicnn.paths import BEST_MODELS_ROOT
from minicnn.schedulers.cosine import CosineAnnealingLR
from minicnn.schedulers.plateau import ReduceLROnPlateau
from minicnn.schedulers.step import StepLR

TRAINING_SUMMARY_SCHEMA_VERSION = 1


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
    loss_map = {'CrossEntropyLoss': 'cross_entropy', 'MSELoss': 'mse'}
    loss_type_str = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    loss_type = loss_map.get(loss_type_str)
    if loss_type is None:
        raise ValueError(
            f'cuda_native does not support loss.type={loss_type_str!r}. '
            'Supported: CrossEntropyLoss, MSELoss.'
        )
    return loss_type


def build_epoch_row(*, epoch: int, train_loss: float, val_metrics: dict[str, float], lr: float, epoch_time_s: float) -> dict[str, Any]:
    return {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_acc': val_metrics['acc'],
        'lr': lr,
        'epoch_time_s': epoch_time_s,
    }


def epoch_log_message(*, epoch: int, epochs: int, train_loss: float, val_acc: float, lr: float, epoch_time_s: float) -> str:
    return (
        f"[cuda_native] Epoch {epoch}/{epochs}: "
        f"train_loss={train_loss:.4f}, "
        f"val_acc={val_acc * 100:.2f}%, "
        f"lr={lr:.6f}, "
        f"time={epoch_time_s:.1f}s"
    )


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
    lr: float,
    momentum: float,
    grad_clip_global: float,
    epochs: int,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    return {
        'schema_version': TRAINING_SUMMARY_SCHEMA_VERSION,
        'artifact_kind': 'training_run_summary',
        'status': 'ok',
        'selected_backend': 'cuda_native',
        'effective_backend': 'cuda_native',
        'variant': 'reference',
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
        'periodic_checkpoints': [],
        'test_loss': None,
        'test_acc': None,
        'capabilities': capabilities,
    }
