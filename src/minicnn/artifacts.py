from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    return {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
    }


def _detect_npz_kind(path: Path, keys: list[str]) -> str:
    key_set = set(keys)
    if {'epoch', 'val_acc', 'fc_w', 'fc_b'} <= key_set:
        return 'cuda_legacy_checkpoint'
    if path.name.endswith('_autograd_best.npz'):
        return 'autograd_state_dict'
    if any(key.startswith('_running_mean_') or key.startswith('_running_var_') for key in key_set):
        return 'cuda_native_param_dict'
    if any(key.startswith('_w_') or key.startswith('_b_') for key in key_set):
        return 'cuda_native_param_dict'
    return 'numpy_state_dict'


def inspect_npz_checkpoint(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with np.load(path_obj) as ckpt:
        keys = sorted(ckpt.files)
        preview = {
            key: _array_summary(ckpt[key])
            for key in keys[:20]
        }
    kind = _detect_npz_kind(path_obj, keys)
    compatibility = {
        'cuda_legacy_checkpoint': ['cuda_legacy'],
        'autograd_state_dict': ['autograd'],
        'cuda_native_param_dict': ['cuda_native'],
        'numpy_state_dict': ['numpy_state_dict_only'],
    }.get(kind, ['unknown'])
    return {
        'path': str(path_obj),
        'format': 'npz',
        'kind': kind,
        'compatible_backends': compatibility,
        'keys': keys,
        'num_keys': len(keys),
        'preview': preview,
    }


def inspect_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised via CLI subprocess
        raise RuntimeError(
            'inspect-checkpoint requires PyTorch to read .pt/.pth files.\n'
            'Install it with:\n'
            '  pip install -e .[torch]'
        ) from exc

    try:
        payload = torch.load(path_obj, map_location='cpu', weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path_obj, map_location='cpu')
    if not isinstance(payload, dict):
        return {
            'path': str(path_obj),
            'format': 'torch_pickle',
            'kind': 'unknown_torch_payload',
            'top_level_type': type(payload).__name__,
        }
    model_state = payload.get('model_state')
    if isinstance(model_state, dict):
        state_keys = sorted(model_state.keys())
        preview = {
            key: _array_summary(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)
            for key, value in list(model_state.items())[:20]
        }
    else:
        state_keys = []
        preview = {}
    return {
        'path': str(path_obj),
        'format': 'pt',
        'kind': 'torch_state_dict_checkpoint',
        'compatible_backends': ['torch', 'train-flex', 'train-dual(engine.backend=torch)'],
        'top_level_keys': sorted(payload.keys()),
        'state_keys': state_keys,
        'num_state_keys': len(state_keys),
        'preview': preview,
    }


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {path_obj}')
    suffix = path_obj.suffix.lower()
    if suffix == '.npz':
        return inspect_npz_checkpoint(path_obj)
    if suffix in {'.pt', '.pth'}:
        return inspect_torch_checkpoint(path_obj)
    raise ValueError(
        f'Unsupported checkpoint format {suffix!r}. '
        'Expected one of: .npz, .pt, .pth'
    )


def _load_config_for_export(config_path: str | Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8')) or {}
    if not isinstance(cfg, dict):
        raise TypeError('Config file must contain a mapping at the top level')
    return cfg


def _build_torch_model_for_export(config_path: str | Path):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            'export-torch-checkpoint requires PyTorch.\n'
            'Install it with:\n'
            '  pip install -e .[torch]'
        ) from exc

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

    for key, target_tensor in target_state.items():
        if key in arrays:
            arr = arrays[key]
            # autograd Linear stores weights as (in_features, out_features)
            if arr.ndim == 2 and tuple(arr.shape[::-1]) == tuple(target_tensor.shape):
                arr = arr.T
            if tuple(arr.shape) != tuple(target_tensor.shape):
                raise ValueError(
                    f'Cannot export autograd checkpoint: key {key!r} has shape {tuple(arr.shape)} '
                    f'but torch model expects {tuple(target_tensor.shape)}'
                )
            converted[key] = torch.from_numpy(np.asarray(arr)).to(dtype=target_tensor.dtype)
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
    torch.save({
        'model_state': converted,
        'source_format': 'autograd_state_dict',
        'source_checkpoint': str(checkpoint_path),
        'config_path': str(config_path),
        'defaulted_keys': defaulted,
        'backend_hint': 'torch',
    }, output)
    return {
        'ok': True,
        'source_format': 'autograd_state_dict',
        'output_path': str(output),
        'defaulted_keys': defaulted,
        'num_keys': len(converted),
        'model_layers': [layer.get('type') for layer in cfg.get('model', {}).get('layers', [])],
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

    if missing:
        raise ValueError(
            'Cannot export cuda_native checkpoint to torch: missing keys '
            f'{missing}. Only sequential configs that map cleanly onto torch are supported.'
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': converted,
        'source_format': 'cuda_native_param_dict',
        'source_checkpoint': str(checkpoint_path),
        'config_path': str(config_path),
        'defaulted_keys': defaulted,
        'backend_hint': 'torch',
    }, output)
    return {
        'ok': True,
        'source_format': 'cuda_native_param_dict',
        'output_path': str(output),
        'defaulted_keys': defaulted,
        'num_keys': len(converted),
        'model_layers': [layer.get('type') for layer in cfg.get('model', {}).get('layers', [])],
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
