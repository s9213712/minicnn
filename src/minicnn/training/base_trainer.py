from __future__ import annotations

import os
from pathlib import Path

from minicnn.config.schema import ExperimentConfig
from minicnn.config.settings import apply_experiment_config
from minicnn.runtime.context import TrainContext
from minicnn.training.callbacks import CallbackList
from minicnn.training.experiment import ExperimentManager
from minicnn.training.profiler import RunProfiler


class BaseTrainer:
    def __init__(self, config: ExperimentConfig, backend, callbacks):
        self.config = config
        self.backend = backend
        self.callbacks = CallbackList(callbacks)
        self.experiment = ExperimentManager(config)
        self.profiler = RunProfiler(enabled=config.runtime.profile)

    def _prepare_runtime(self) -> TrainContext:
        apply_experiment_config(self.config)
        os.environ['MINICNN_ARTIFACT_RUN_DIR'] = str(self.experiment.run_dir)
        return TrainContext(
            config=self.config,
            run_dir=self.experiment.run_dir,
            backend_name=self.backend.name,
            profiler_enabled=self.config.runtime.profile,
        )

    def fit(self) -> Path:
        context = self._prepare_runtime()
        self.experiment.write_metadata()
        self.callbacks.on_fit_start(context)
        self.profiler.start()
        try:
            self.callbacks.on_backend_start(context)
            result = self.backend.run(context)
            self.profiler.mark('backend_finished')
            result['profiler'] = self.profiler.finish()
            self.callbacks.on_backend_end(context, result)
            self.callbacks.on_fit_end(context)
        except BaseException as exc:
            self.callbacks.on_exception(context, exc)
            context.summary.update({'status': 'failed', 'error': str(exc), 'backend': self.backend.name})
            self.callbacks.on_fit_end(context)
            raise
        return self.experiment.run_dir
