"""Public API surface for cuda_native."""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import (
    CUDA_NATIVE_CAPABILITIES,
    get_cuda_native_capabilities,
)
from minicnn.cuda_native.graph import NativeGraph, build_graph
from minicnn.cuda_native.validators import validate_cuda_native_model_config


_SUPPORTED_DATASET_TYPES = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_datasets']
)
_SUPPORTED_LOSS_TYPES = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_losses']
)
_SUPPORTED_OPTIMIZERS = frozenset(
    str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_optimizers']
)


def _as_mapping(name: str, value: Any) -> tuple[dict[str, Any], list[str]]:
    if value is None:
        return {}, []
    if isinstance(value, dict):
        return value, []
    return {}, [f'{name} must be a mapping']


def _coerce_float(name: str, value: Any) -> tuple[float | None, list[str]]:
    try:
        return float(value), []
    except (TypeError, ValueError):
        return None, [f'{name} must be numeric, got {value!r}']


def _coerce_int(name: str, value: Any) -> tuple[int | None, list[str]]:
    try:
        return int(value), []
    except (TypeError, ValueError):
        return None, [f'{name} must be an integer, got {value!r}']


def _validate_dataset_cfg(dataset_cfg: dict[str, Any]) -> list[str]:
    dtype = str(dataset_cfg.get('type', 'random'))
    if dtype not in _SUPPORTED_DATASET_TYPES:
        supported = ', '.join(sorted(_SUPPORTED_DATASET_TYPES))
        return [
            f'cuda_native does not support dataset.type={dtype!r}. Supported: {supported}.'
        ]
    return []


def _validate_loss_cfg(loss_cfg: dict[str, Any]) -> list[str]:
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    if loss_type not in _SUPPORTED_LOSS_TYPES:
        supported = ', '.join(sorted(_SUPPORTED_LOSS_TYPES))
        return [
            f'cuda_native does not support loss.type={loss_type!r}. Supported: {supported}.'
        ]
    return []


def _validate_optimizer_cfg(optim_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    opt_type = str(optim_cfg.get('type', 'SGD'))
    if opt_type not in _SUPPORTED_OPTIMIZERS:
        supported = ', '.join(sorted(_SUPPORTED_OPTIMIZERS))
        errors.append(
            f'cuda_native only supports optimizer.type in {{{supported}}}; got {opt_type!r}.'
        )
        return errors

    base_lr, lr_errors = _coerce_float('optimizer.lr', optim_cfg.get('lr', 0.01))
    errors.extend(lr_errors)

    momentum_val, momentum_errors = _coerce_float(
        'optimizer.momentum',
        optim_cfg.get('momentum', 0.0),
    )
    errors.extend(momentum_errors)
    if momentum_val is not None and momentum_val != 0.0:
        errors.append(
            'cuda_native training uses plain SGD only; set optimizer.momentum=0 '
            'or use engine.backend=torch.'
        )

    grad_clip_val, grad_clip_errors = _coerce_float(
        'optimizer.grad_clip_global',
        optim_cfg.get('grad_clip_global', 0.0),
    )
    errors.extend(grad_clip_errors)
    if grad_clip_val is not None and grad_clip_val != 0.0:
        errors.append(
            'cuda_native does not support optimizer.grad_clip_global; set it to 0.'
        )

    for field in ('lr_conv1', 'lr_conv', 'lr_fc'):
        if field not in optim_cfg:
            continue
        field_lr, field_errors = _coerce_float(f'optimizer.{field}', optim_cfg[field])
        errors.extend(field_errors)
        if base_lr is not None and field_lr is not None and field_lr != base_lr:
            errors.append(
                f'cuda_native uses a single optimizer.lr; optimizer.{field}={field_lr} '
                f'must match optimizer.lr={base_lr}.'
            )
    return errors


def _validate_scheduler_cfg(scheduler_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    enabled = bool(scheduler_cfg.get('enabled', False))
    scheduler_type = str(scheduler_cfg.get('type', 'none') or 'none')
    normalized = scheduler_type.lower()
    no_scheduler = normalized in {'none', 'disabled', 'null', ''}
    if enabled and no_scheduler:
        errors.append(
            'cuda_native expects scheduler.enabled=false when no scheduler is used.'
        )
    elif enabled and not no_scheduler:
        errors.append(
            'cuda_native does not support schedulers yet; set scheduler.enabled=false '
            f'(got scheduler.type={scheduler_type!r}).'
        )
    return errors


def _validate_train_cfg(train_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if bool(train_cfg.get('amp', False)):
        errors.append('cuda_native does not support train.amp=true.')
    grad_accum_steps, grad_accum_errors = _coerce_int(
        'train.grad_accum_steps',
        train_cfg.get('grad_accum_steps', 1),
    )
    errors.extend(grad_accum_errors)
    if grad_accum_steps is not None and grad_accum_steps != 1:
        errors.append(
            'cuda_native does not support train.grad_accum_steps other than 1.'
        )
    device = train_cfg.get('device')
    if device not in {None, 'auto', 'cpu'}:
        errors.append(
            f'cuda_native ignores train.device={device!r}; use "auto" or "cpu".'
        )
    return errors


def validate_cuda_native_config(cfg: dict[str, Any]) -> list[str]:
    """Validate a full experiment config for cuda_native compatibility.

    Returns a list of error strings (empty list = valid).
    """
    errors: list[str] = []

    model_cfg, model_cfg_errors = _as_mapping('model', cfg.get('model'))
    dataset_cfg, dataset_cfg_errors = _as_mapping('dataset', cfg.get('dataset'))
    loss_cfg, loss_cfg_errors = _as_mapping('loss', cfg.get('loss'))
    optim_cfg, optim_cfg_errors = _as_mapping('optimizer', cfg.get('optimizer'))
    scheduler_cfg, scheduler_cfg_errors = _as_mapping('scheduler', cfg.get('scheduler'))
    train_cfg, train_cfg_errors = _as_mapping('train', cfg.get('train'))

    errors.extend(model_cfg_errors)
    errors.extend(dataset_cfg_errors)
    errors.extend(loss_cfg_errors)
    errors.extend(optim_cfg_errors)
    errors.extend(scheduler_cfg_errors)
    errors.extend(train_cfg_errors)

    errors.extend(validate_cuda_native_model_config(model_cfg))
    errors.extend(_validate_dataset_cfg(dataset_cfg))
    errors.extend(_validate_loss_cfg(loss_cfg))
    errors.extend(_validate_optimizer_cfg(optim_cfg))
    errors.extend(_validate_scheduler_cfg(scheduler_cfg))
    errors.extend(_validate_train_cfg(train_cfg))

    layers = model_cfg.get('layers', [])
    if any(str(layer.get('type')) == 'BatchNorm2d' for layer in layers):
        errors.append(
            'cuda_native training path does not yet support BatchNorm2d backward. '
            'BatchNorm2d forward exists, but train-native still rejects BN-containing models.'
        )
    return errors


def build_cuda_native_graph(
    model_cfg: dict[str, Any],
    input_shape: tuple[int, ...],
) -> NativeGraph:
    """Build and return a NativeGraph from a model config dict.

    Args:
        model_cfg:   dict with a 'layers' key (same format as flex/autograd).
        input_shape: fixed input shape, e.g. (1, 3, 32, 32).

    Raises:
        ValueError: if the config references unsupported ops or has bad attrs/shapes.
    """
    layers = model_cfg.get('layers', [])
    return build_graph(layers, input_shape)


def get_capability_summary() -> dict[str, object]:
    """Return the cuda_native capability summary for diagnostics."""
    return get_cuda_native_capabilities()
