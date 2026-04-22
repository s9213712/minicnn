from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minicnn._artifact_inspect import inspect_checkpoint
from minicnn.checkpoint_schema import (
    TORCH_STATE_DICT_CHECKPOINT_KIND,
    build_checkpoint_info,
    checkpoint_fingerprint,
)
from minicnn.torch_runtime import require_torch


def _load_config_for_export(config_path: str | Path) -> dict[str, Any]:
    from minicnn.flex.config import load_flex_config

    return load_flex_config(config_path)


def _build_torch_model_for_export(config_path: str | Path):
    torch = require_torch(
        'export-torch-checkpoint',
        action='export MiniCNN checkpoints to torch',
    )

    from minicnn.flex.builder import build_model

    cfg = _load_config_for_export(config_path)
    dataset_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})
    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    model = build_model(model_cfg, input_shape=input_shape)
    return cfg, model, torch


def _export_autograd_npz_to_torch(
    checkpoint_path: Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    cfg, model, torch = _build_torch_model_for_export(config_path)
    with np.load(checkpoint_path) as ckpt:
        arrays = {k: ckpt[k] for k in ckpt.files}

    target_state = model.state_dict()
    converted: dict[str, Any] = {}
    defaulted: list[str] = []
    missing: list[str] = []
    transposed: list[str] = []
    consumed_source_keys: set[str] = set()

    for key, target_tensor in target_state.items():
        if key in arrays:
            arr = arrays[key]
            if arr.ndim == 2 and tuple(arr.shape[::-1]) == tuple(target_tensor.shape):
                arr = arr.T
                transposed.append(key)
            if tuple(arr.shape) != tuple(target_tensor.shape):
                raise ValueError(
                    f'Cannot export autograd checkpoint: key {key!r} has shape {tuple(arr.shape)} '
                    f'but torch model expects {tuple(target_tensor.shape)}'
                )
            converted[key] = torch.from_numpy(np.asarray(arr)).to(dtype=target_tensor.dtype)
            consumed_source_keys.add(key)
        elif key.endswith('running_mean'):
            converted[key] = torch.zeros_like(target_tensor)
            defaulted.append(key)
        elif key.endswith('running_var'):
            converted[key] = torch.ones_like(target_tensor)
            defaulted.append(key)
        elif key.endswith('num_batches_tracked'):
            converted[key] = torch.zeros_like(target_tensor)
            defaulted.append(key)
        else:
            missing.append(key)

    if missing:
        raise ValueError(
            'Cannot export autograd checkpoint to torch: missing keys '
            f'{missing}. Use the same architecture config that produced the checkpoint.'
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    conversion_report = {
        'transposed_keys': transposed,
        'defaulted_keys': defaulted,
        'skipped_source_keys': sorted(set(arrays) - consumed_source_keys),
        'source_checkpoint_fingerprint': checkpoint_fingerprint(checkpoint_path),
    }
    torch.save({
        'model_state': converted,
        'source_format': 'autograd_state_dict',
        'source_checkpoint': str(checkpoint_path),
        'config_path': str(config_path),
        'defaulted_keys': defaulted,
        'backend_hint': 'torch',
        'conversion_report': conversion_report,
    }, output)
    output_info = build_checkpoint_info(
        path=output,
        format='pt',
        kind=TORCH_STATE_DICT_CHECKPOINT_KIND,
        warnings=[],
    )
    return {
        **output_info.to_dict(),
        'ok': True,
        'source_format': 'autograd_state_dict',
        'output_path': str(output),
        'defaulted_keys': defaulted,
        'num_keys': len(converted),
        'model_layers': [layer.get('type') for layer in cfg.get('model', {}).get('layers', [])],
        'metadata': {
            'source_checkpoint': str(checkpoint_path),
            'config_path': str(config_path),
            'backend_hint': 'torch',
        },
        'conversion_report': conversion_report,
    }


def _export_cuda_native_npz_to_torch(
    checkpoint_path: Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    cfg, model, torch = _build_torch_model_for_export(config_path)
    with np.load(checkpoint_path) as ckpt:
        arrays = {k: ckpt[k] for k in ckpt.files}

    target_state = model.state_dict()
    layers = cfg.get('model', {}).get('layers', [])
    converted: dict[str, Any] = {}
    missing: list[str] = []
    defaulted: list[str] = []
    consumed_source_keys: set[str] = set()

    def source_key(idx: int, suffix: str) -> str:
        op = str(layers[idx]['type']).lower()
        return f'{suffix}_{op}_{idx}'

    for key, target_tensor in target_state.items():
        prefix, field = key.rsplit('.', 1)
        if not prefix.isdigit():
            missing.append(key)
            continue
        idx = int(prefix)
        if idx >= len(layers):
            missing.append(key)
            continue
        if field == 'weight':
            src = source_key(idx, '_w')
        elif field == 'bias':
            src = source_key(idx, '_b')
        elif field == 'running_mean':
            src = source_key(idx, '_running_mean')
        elif field == 'running_var':
            src = source_key(idx, '_running_var')
        elif field == 'num_batches_tracked':
            converted[key] = torch.zeros_like(target_tensor)
            defaulted.append(key)
            continue
        else:
            missing.append(key)
            continue

        if src not in arrays:
            if field == 'bias':
                converted[key] = torch.zeros_like(target_tensor)
                defaulted.append(key)
                continue
            missing.append(key)
            continue
        arr = arrays[src]
        if tuple(arr.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f'Cannot export cuda_native checkpoint: key {src!r} has shape {tuple(arr.shape)} '
                f'but torch model expects {tuple(target_tensor.shape)} for {key!r}'
            )
        converted[key] = torch.from_numpy(np.asarray(arr)).to(dtype=target_tensor.dtype)
        consumed_source_keys.add(src)

    if missing:
        raise ValueError(
            'Cannot export cuda_native checkpoint to torch: missing keys '
            f'{missing}. Only sequential configs that map cleanly onto torch are supported.'
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    conversion_report = {
        'transposed_keys': [],
        'defaulted_keys': defaulted,
        'skipped_source_keys': sorted(set(arrays) - consumed_source_keys),
        'source_checkpoint_fingerprint': checkpoint_fingerprint(checkpoint_path),
    }
    torch.save({
        'model_state': converted,
        'source_format': 'cuda_native_param_dict',
        'source_checkpoint': str(checkpoint_path),
        'config_path': str(config_path),
        'defaulted_keys': defaulted,
        'backend_hint': 'torch',
        'conversion_report': conversion_report,
    }, output)
    output_info = build_checkpoint_info(
        path=output,
        format='pt',
        kind=TORCH_STATE_DICT_CHECKPOINT_KIND,
        warnings=[],
    )
    return {
        **output_info.to_dict(),
        'ok': True,
        'source_format': 'cuda_native_param_dict',
        'output_path': str(output),
        'defaulted_keys': defaulted,
        'num_keys': len(converted),
        'model_layers': [layer.get('type') for layer in cfg.get('model', {}).get('layers', [])],
        'metadata': {
            'source_checkpoint': str(checkpoint_path),
            'config_path': str(config_path),
            'backend_hint': 'torch',
        },
        'conversion_report': conversion_report,
    }


def export_checkpoint_to_torch(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    path_obj = Path(checkpoint_path)
    meta = inspect_checkpoint(path_obj)
    kind = str(meta.get('kind', ''))
    if kind == 'autograd_state_dict':
        return _export_autograd_npz_to_torch(path_obj, config_path, output_path)
    if kind == 'cuda_native_param_dict':
        return _export_cuda_native_npz_to_torch(path_obj, config_path, output_path)
    if kind == 'cuda_legacy_checkpoint':
        raise ValueError(
            'cuda_legacy checkpoints cannot be exported directly to a generic torch checkpoint.\n'
            'They are tied to the handcrafted CUDA runtime geometry and checkpoint schema.'
        )
    if str(meta.get('format')) == 'pt':
        raise ValueError('This checkpoint is already a torch-format file.')
    raise ValueError(
        f'Checkpoint kind {kind!r} is not currently exportable to torch. '
        'Supported sources: autograd_state_dict, cuda_native_param_dict.'
    )
