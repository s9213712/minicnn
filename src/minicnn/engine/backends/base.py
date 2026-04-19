from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendCapabilities:
    amp: bool = False
    grad_accumulation: bool = False
    fine_grained_callbacks: bool = False
    profiler: bool = False
    module_system: bool = False
    optimizer_stack: bool = False


class BaseBackend:
    name = 'base'
    capabilities = BackendCapabilities()

    def __init__(self, framework_stack: dict[str, Any] | None = None):
        self.framework_stack = framework_stack or {}

    def describe_framework_stack(self) -> dict[str, Any]:
        optimizer = self.framework_stack.get('optimizer')
        scheduler = self.framework_stack.get('scheduler')
        model = self.framework_stack.get('model')
        return {
            'has_model': model is not None,
            'optimizer': optimizer.__class__.__name__ if optimizer else None,
            'scheduler': scheduler.__class__.__name__ if scheduler else None,
            'num_parameters': len(model.parameters()) if model is not None else 0,
        }

    def run(self, context) -> dict[str, Any]:
        raise NotImplementedError
