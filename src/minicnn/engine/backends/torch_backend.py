from __future__ import annotations

import os
from typing import Any

from .base import BackendCapabilities, BaseBackend


class TorchLegacyBackend(BaseBackend):
    name = 'torch'
    capabilities = BackendCapabilities(
        amp=True,
        grad_accumulation=True,
        fine_grained_callbacks=False,
        profiler=True,
        module_system=True,
        optimizer_stack=True,
    )

    def run(self, context) -> dict[str, Any]:
        os.environ['MINICNN_V5_TRAINER'] = 'torch'
        from minicnn.training.train_torch_baseline import main as entry
        entry()
        return {
            'status': 'completed',
            'backend': self.name,
            'mode': 'legacy-adapter',
            'supported': {'amp': True, 'grad_accumulation': True},
            'framework_stack': self.describe_framework_stack(),
        }
