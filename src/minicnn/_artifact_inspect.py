from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minicnn.checkpoint_schema import (
    TORCH_STATE_DICT_CHECKPOINT_KIND,
    UNKNOWN_TORCH_PAYLOAD_KIND,
    build_checkpoint_info,
    detect_npz_kind,
    warnings_for_kind,
)
from minicnn.torch_runtime import require_torch


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    return {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
    }


def inspect_npz_checkpoint(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with np.load(path_obj) as ckpt:
        keys = sorted(ckpt.files)
        preview = {
            key: _array_summary(ckpt[key])
            for key in keys[:20]
        }
        metadata = {
            key: ckpt[key].tolist()
            for key in ('schema_version', 'backend', 'checkpoint_kind', 'created_at', 'epoch', 'val_acc')
            if key in ckpt
        }
    kind = detect_npz_kind(path_obj, keys)
    info = build_checkpoint_info(path=path_obj, format='npz', kind=kind, warnings=warnings_for_kind(kind))
    payload = {
        **info.to_dict(),
        'keys': keys,
        'num_keys': len(keys),
        'preview': preview,
    }
    if metadata:
        payload['metadata'] = metadata
    return payload


def inspect_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    torch = require_torch('inspect-checkpoint', action='read .pt/.pth files')

    try:
        payload = torch.load(path_obj, map_location='cpu', weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path_obj, map_location='cpu')
    if not isinstance(payload, dict):
        info = build_checkpoint_info(
            path=path_obj,
            format='torch_pickle',
            kind=UNKNOWN_TORCH_PAYLOAD_KIND,
            warnings=['Torch payload is not a dictionary; inspection is limited.'],
        )
        return {
            **info.to_dict(),
            'top_level_type': type(payload).__name__,
            'preview': {},
        }
    model_state = payload.get('model_state')
    metadata = {
        key: payload[key]
        for key in ('source_format', 'source_checkpoint', 'config_path', 'backend_hint', 'defaulted_keys', 'conversion_report')
        if key in payload
    }
    warnings: list[str] = []
    if isinstance(model_state, dict):
        state_keys = sorted(model_state.keys())
        preview = {
            key: _array_summary(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)
            for key, value in list(model_state.items())[:20]
        }
    else:
        state_keys = []
        preview = {}
        warnings.append("Torch checkpoint does not contain a 'model_state' dictionary.")
    info = build_checkpoint_info(
        path=path_obj,
        format='pt',
        kind=TORCH_STATE_DICT_CHECKPOINT_KIND,
        warnings=warnings,
    )
    result = {
        **info.to_dict(),
        'top_level_keys': sorted(payload.keys()),
        'state_keys': state_keys,
        'num_state_keys': len(state_keys),
        'preview': preview,
    }
    if metadata:
        result['metadata'] = metadata
    return result


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
