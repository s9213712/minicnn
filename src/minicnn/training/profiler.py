from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunProfiler:
    enabled: bool = False
    _t0: float | None = None
    marks: dict[str, float] = field(default_factory=dict)

    def start(self):
        if self.enabled:
            self._t0 = time.perf_counter()

    def mark(self, name: str):
        if self.enabled and self._t0 is not None:
            self.marks[name] = time.perf_counter() - self._t0

    def finish(self) -> dict[str, Any]:
        if not self.enabled or self._t0 is None:
            return {'enabled': False}
        total = time.perf_counter() - self._t0
        return {'enabled': True, 'total_seconds': total, 'marks': dict(self.marks)}
