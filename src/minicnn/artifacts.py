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
