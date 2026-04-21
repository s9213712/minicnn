from __future__ import annotations

from .config import load_flex_config

__all__ = [
    'load_flex_config',
    'build_model',
    'build_loss',
    'build_optimizer',
    'build_scheduler',
]


def __getattr__(name: str):
    if name in {'build_model', 'build_loss', 'build_optimizer', 'build_scheduler'}:
        from .builder import build_loss, build_model, build_optimizer, build_scheduler

        exports = {
            'build_model': build_model,
            'build_loss': build_loss,
            'build_optimizer': build_optimizer,
            'build_scheduler': build_scheduler,
        }
        return exports[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
