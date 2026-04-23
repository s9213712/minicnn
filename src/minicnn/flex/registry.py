from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class Registry:
    def __init__(self):
        self._items: dict[str, dict[str, Callable[..., Any]]] = defaultdict(dict)

    def register(
        self,
        category: str,
        name: str,
        factory: Callable[..., Any],
        *,
        replace: bool = False,
    ):
        if not replace and name in self._items.get(category, {}):
            raise ValueError(
                f"Registry entry already exists for {category!r}/{name!r}; "
                "pass replace=True to overwrite it explicitly"
            )
        self._items[category][name] = factory

    def get(self, category: str, name: str) -> Callable[..., Any]:
        return self._items[category][name]

    def has(self, category: str, name: str) -> bool:
        return name in self._items.get(category, {})

    def summary(self) -> dict[str, list[str]]:
        return {category: sorted(values) for category, values in sorted(self._items.items())}


REGISTRY = Registry()


def register(category: str, name: str, *, replace: bool = False):
    def wrapper(factory: Callable[..., Any]):
        REGISTRY.register(category, name, factory, replace=replace)
        return factory
    return wrapper


def describe_registries() -> dict[str, list[str]]:
    from . import components  # noqa: F401  # force registration
    return REGISTRY.summary()
