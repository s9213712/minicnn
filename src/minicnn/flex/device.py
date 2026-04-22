from __future__ import annotations

from minicnn.torch_runtime import import_torch_with_details, resolve_torch_device


torch, _TORCH_IMPORT_ERROR = import_torch_with_details()


def _choose_device(device_cfg: str):
    if torch is None:
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                'PyTorch import failed in this environment. '
                f'{_TORCH_IMPORT_ERROR.__class__.__name__}: {_TORCH_IMPORT_ERROR}'
            )
        raise RuntimeError('PyTorch is required for train-flex.')
    return resolve_torch_device(device_cfg, torch)
