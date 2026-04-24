from __future__ import annotations

import os
import sys
import warnings


SUPPRESS_EXPERIMENTAL_WARN_ENV = 'MINICNN_SUPPRESS_EXPERIMENTAL_WARN'


def should_emit_experimental_warning() -> bool:
    if os.environ.get(SUPPRESS_EXPERIMENTAL_WARN_ENV, '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        return False
    stdout = getattr(sys, 'stdout', None)
    isatty = getattr(stdout, 'isatty', None)
    if callable(isatty):
        try:
            return bool(isatty())
        except Exception:
            return False
    return False


def emit_experimental_warning(message: str, *, stacklevel: int = 2) -> None:
    if not should_emit_experimental_warning():
        return
    warnings.warn(message, UserWarning, stacklevel=stacklevel)
