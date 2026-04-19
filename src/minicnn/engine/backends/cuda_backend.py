from __future__ import annotations

import os
from typing import Any

from .base import BackendCapabilities, BaseBackend


class CudaLegacyBackend(BaseBackend):
    name = 'cuda'
    capabilities = BackendCapabilities(
        amp=False,
        grad_accumulation=False,
        fine_grained_callbacks=False,
        profiler=True,
        module_system=True,
        optimizer_stack=True,
    )

    def run(self, context) -> dict[str, Any]:
        cfg = context.config
        requested = {
            'amp': cfg.runtime.amp,
            'grad_accum_steps': cfg.train.grad_accum_steps,
        }
        unsupported = []
        if cfg.runtime.amp:
            unsupported.append('runtime.amp')
        if cfg.train.grad_accum_steps != 1:
            unsupported.append('train.grad_accum_steps')
        if unsupported:
            print(f"[minicnn] cuda legacy backend ignores unsupported V5 features: {', '.join(unsupported)}")
        os.environ['MINICNN_V5_TRAINER'] = 'cuda'
        from minicnn.training.train_cuda import main as entry
        entry()
        return {
            'status': 'completed',
            'backend': self.name,
            'mode': 'legacy-adapter',
            'requested': requested,
            'supported': {'amp': False, 'grad_accumulation': False},
            'framework_stack': self.describe_framework_stack(),
        }
