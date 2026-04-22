from __future__ import annotations

from typing import Any

import yaml


def parse_bool(value: Any, *, label: str = 'value') -> bool:
    accepted = 'accepted values: true, false, yes, no, on, off, 1, 0'
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    raise ValueError(f'{label} must be a boolean, got {value!r}; {accepted}')


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


def set_nested_value(
    data: dict[str, Any],
    parts: list[str],
    value: Any,
    *,
    clear_on_type_change: bool = False,
) -> None:
    if not parts:
        raise ValueError('Override key cannot be empty')

    cur: Any = data
    for depth, part in enumerate(parts[:-1]):
        next_part = parts[depth + 1]
        if isinstance(cur, list):
            try:
                index = int(part)
            except ValueError as exc:
                path = '.'.join(parts[: depth + 1])
                raise TypeError(f'Override path {path!r} must use a numeric list index') from exc
            try:
                cur = cur[index]
            except IndexError as exc:
                path = '.'.join(parts[: depth + 1])
                raise IndexError(f'Override path {path!r} is out of range') from exc
        elif isinstance(cur, dict):
            if part not in cur:
                cur[part] = [] if next_part.isdigit() else {}
            cur = cur[part]
        else:
            path = '.'.join(parts[:depth])
            raise TypeError(f'Override path {path!r} does not refer to a container')

    last = parts[-1]
    if isinstance(cur, list):
        try:
            index = int(last)
        except ValueError as exc:
            path = '.'.join(parts)
            raise TypeError(f'Override path {path!r} must end with a numeric list index') from exc
        try:
            cur[index] = value
        except IndexError as exc:
            path = '.'.join(parts)
            raise IndexError(f'Override path {path!r} is out of range') from exc
        return

    if not isinstance(cur, dict):
        path = '.'.join(parts[:-1])
        raise TypeError(f'Override path {path!r} does not refer to a mapping')
    if clear_on_type_change and last == 'type' and cur.get('type') != value:
        cur.clear()
    cur[last] = value
