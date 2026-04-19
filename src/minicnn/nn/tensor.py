from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tensor:
    data: Any
    grad: Any = None
    requires_grad: bool = False
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def zero_grad(self):
        self.grad = None
