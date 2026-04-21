from __future__ import annotations

from .config import dump_unified_template, load_unified_config

__all__ = [
    'load_unified_config',
    'dump_unified_template',
    'train_unified_from_config',
    'validate_cuda_legacy_compatibility',
    'compile_to_legacy_experiment',
]


def __getattr__(name: str):
    if name == 'train_unified_from_config':
        from .trainer import train_unified_from_config

        return train_unified_from_config
    if name in {'validate_cuda_legacy_compatibility', 'compile_to_legacy_experiment'}:
        from .cuda_legacy import compile_to_legacy_experiment, validate_cuda_legacy_compatibility

        exports = {
            'validate_cuda_legacy_compatibility': validate_cuda_legacy_compatibility,
            'compile_to_legacy_experiment': compile_to_legacy_experiment,
        }
        return exports[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
