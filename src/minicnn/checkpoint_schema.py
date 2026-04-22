from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1
CUDA_LEGACY_CHECKPOINT_KIND = 'cuda_legacy_checkpoint'
AUTOGRAD_STATE_DICT_KIND = 'autograd_state_dict'
CUDA_NATIVE_PARAM_DICT_KIND = 'cuda_native_param_dict'
NUMPY_STATE_DICT_KIND = 'numpy_state_dict'
TORCH_STATE_DICT_CHECKPOINT_KIND = 'torch_state_dict_checkpoint'
UNKNOWN_TORCH_PAYLOAD_KIND = 'unknown_torch_payload'


@dataclass
class CheckpointInfo:
    schema_version: int
    path: str
    format: str
    kind: str
    fingerprint: str
    compatible_backends: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def detect_npz_kind(path: Path, keys: list[str]) -> str:
    key_set = set(keys)
    if {'epoch', 'val_acc', 'fc_w', 'fc_b'} <= key_set:
        return CUDA_LEGACY_CHECKPOINT_KIND
    if path.name.endswith('_autograd_best.npz'):
        return AUTOGRAD_STATE_DICT_KIND
    if any(key.startswith('_running_mean_') or key.startswith('_running_var_') for key in key_set):
        return CUDA_NATIVE_PARAM_DICT_KIND
    if any(key.startswith('_w_') or key.startswith('_b_') for key in key_set):
        return CUDA_NATIVE_PARAM_DICT_KIND
    return NUMPY_STATE_DICT_KIND


def compatible_backends_for_kind(kind: str) -> list[str]:
    return {
        CUDA_LEGACY_CHECKPOINT_KIND: ['cuda_legacy'],
        AUTOGRAD_STATE_DICT_KIND: ['autograd'],
        CUDA_NATIVE_PARAM_DICT_KIND: ['cuda_native'],
        NUMPY_STATE_DICT_KIND: ['numpy_state_dict_only'],
        TORCH_STATE_DICT_CHECKPOINT_KIND: ['torch', 'train-flex', 'train-dual(engine.backend=torch)'],
    }.get(kind, ['unknown'])


def warnings_for_kind(kind: str) -> list[str]:
    warnings: list[str] = []
    if kind == CUDA_LEGACY_CHECKPOINT_KIND:
        warnings.append(
            'This checkpoint is tied to the handcrafted cuda_legacy runtime schema and cannot be exported directly to torch.'
        )
    elif kind == NUMPY_STATE_DICT_KIND:
        warnings.append(
            'This NPZ file does not match a recognized MiniCNN training artifact schema.'
        )
    return warnings


def build_checkpoint_info(*, path: str | Path, format: str, kind: str, warnings: list[str] | None = None) -> CheckpointInfo:
    return CheckpointInfo(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        path=str(path),
        format=format,
        kind=kind,
        fingerprint=checkpoint_fingerprint(path),
        compatible_backends=compatible_backends_for_kind(kind),
        warnings=list(warnings or []),
    )
