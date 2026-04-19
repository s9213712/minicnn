from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class Profiler:
    def __init__(self):
        self.records: list[dict[str, float | str]] = []

    @contextmanager
    def record(self, name: str):
        t0 = time.perf_counter()
        yield
        self.records.append({'name': name, 'duration_s': time.perf_counter() - t0})

    def summary(self):
        return {'events': self.records, 'total_s': sum(float(r['duration_s']) for r in self.records)}

    def dump_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding='utf-8')
