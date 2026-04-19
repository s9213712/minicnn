from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    category: str
    factory: Callable[..., Any]
    description: str = ''


class Registry:
    def __init__(self):
        self._items: dict[str, dict[str, ComponentSpec]] = {}

    def register(self, category: str, name: str, factory: Callable[..., Any], description: str = ''):
        bucket = self._items.setdefault(category, {})
        bucket[name] = ComponentSpec(name=name, category=category, factory=factory, description=description)

    def create(self, category: str, name: str, *args, **kwargs):
        return self.get(category, name).factory(*args, **kwargs)

    def get(self, category: str, name: str) -> ComponentSpec:
        try:
            return self._items[category][name]
        except KeyError as exc:
            raise KeyError(f'Unknown component: {category}:{name}') from exc

    def list_category(self, category: str) -> list[ComponentSpec]:
        return [self._items.get(category, {}).get(k) for k in sorted(self._items.get(category, {}))]

    def summary(self) -> dict[str, list[str]]:
        return {category: sorted(items) for category, items in sorted(self._items.items())}


GLOBAL_REGISTRY = Registry()
