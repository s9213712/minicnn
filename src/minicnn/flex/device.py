from __future__ import annotations


try:
    import torch
    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = None if exc.name == 'torch' else exc
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc


def _choose_device(device_cfg: str):
    if torch is None:
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                'PyTorch import failed in this environment. '
                f'{_TORCH_IMPORT_ERROR.__class__.__name__}: {_TORCH_IMPORT_ERROR}'
            )
        raise RuntimeError('PyTorch is required for train-flex.')
    if device_cfg == 'cpu':
        return torch.device('cpu')
    if device_cfg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Requested train.device=cuda, but CUDA is not available in this PyTorch runtime.\n'
                'Use train.device=auto or train.device=cpu.'
            )
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
