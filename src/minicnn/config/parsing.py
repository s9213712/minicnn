from __future__ import annotations

from typing import Any

import yaml


def parse_scalar(text: str) -> Any:
    low = text.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    if low in {'none', 'null'}:
        return None
    try:
        if text.startswith('0') and text not in {'0', '0.0'} and not text.startswith('0.'):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith('[') and text.endswith(']'):
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, list):
            return loaded
    return text
