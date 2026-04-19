from __future__ import annotations

import importlib
from typing import Any


def import_from_string(path: str) -> Any:
    if ':' in path:
        module_name, attr = path.split(':', 1)
    else:
        module_name, attr = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)
