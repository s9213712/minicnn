from __future__ import annotations

import json
from typing import Any, Iterable

from minicnn.utils import write_json


class Callback:
    def on_fit_start(self, context):
        pass

    def on_fit_end(self, context):
        pass

    def on_backend_start(self, context):
        pass

    def on_backend_end(self, context, result: dict[str, Any]):
        pass

    def on_exception(self, context, exc: BaseException):
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: Iterable[Callback]):
        self.callbacks = list(callbacks)

    def on_fit_start(self, context):
        for cb in self.callbacks:
            cb.on_fit_start(context)

    def on_fit_end(self, context):
        for cb in self.callbacks:
            cb.on_fit_end(context)

    def on_backend_start(self, context):
        for cb in self.callbacks:
            cb.on_backend_start(context)

    def on_backend_end(self, context, result: dict[str, Any]):
        for cb in self.callbacks:
            cb.on_backend_end(context, result)

    def on_exception(self, context, exc: BaseException):
        for cb in self.callbacks:
            cb.on_exception(context, exc)


class ConsoleLogger(Callback):
    def on_fit_start(self, context):
        print(f"[minicnn] run_dir={context.run_dir}")
        print(
            f"[minicnn] backend={context.backend_name} "
            f"amp={context.config.runtime.amp} "
            f"grad_accum_steps={context.config.train.grad_accum_steps}"
        )

    def on_backend_start(self, context):
        print('[minicnn] backend run started')

    def on_backend_end(self, context, result: dict[str, Any]):
        print(f"[minicnn] backend run finished: {result}")

    def on_exception(self, context, exc: BaseException):
        print(f"[minicnn] run failed: {exc}")


class SummaryCallback(Callback):
    def on_backend_end(self, context, result: dict[str, Any]):
        context.summary.update(result)
        context.summary['backend'] = context.backend_name
        context.summary['run_dir'] = str(context.run_dir)

    def on_fit_end(self, context):
        write_json(context.run_dir / 'summary.json', context.summary)


class JsonlMetricsCallback(Callback):
    def on_backend_end(self, context, result: dict[str, Any]):
        metrics_path = context.run_dir / context.config.logging.metrics_filename
        with metrics_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


class CheckpointManifestCallback(Callback):
    def on_fit_end(self, context):
        manifest = []
        for p in sorted(context.run_dir.glob('*')):
            manifest.append({'name': p.name, 'is_dir': p.is_dir()})
        write_json(context.run_dir / 'artifacts_manifest.json', {'files': manifest})


class FrameworkManifestCallback(Callback):
    def on_backend_end(self, context, result: dict[str, Any]):
        framework = {
            'backend': context.backend_name,
            'framework': result.get('framework_stack', {}),
            'config': {
                'scheduler_enabled': context.config.scheduler.enabled,
                'scheduler_type': context.config.scheduler.type,
                'module_system': context.config.framework.module_system,
            },
        }
        write_json(context.run_dir / 'framework_manifest.json', framework)
