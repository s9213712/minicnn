from __future__ import annotations

import time
from typing import Any

from minicnn.config.parsing import parse_bool

from .builder import build_loss, build_model, build_optimizer, build_scheduler
from .device import _choose_device, torch
from .reporting import (
    _build_epoch_row,
    _checkpoint_path,
    _epoch_log_message,
    _write_metrics_row,
)
from ._training_steps import (
    adapt_targets as _adapt_targets_impl,
    evaluate_model as _eval_impl,
    pred_accuracy as _pred_accuracy_impl,
    run_train_epoch,
)
from ._training_run import (
    TrainingRunState,
    handle_epoch_artifacts,
    load_best_model_state,
    should_stop_early,
    step_scheduler,
)
from ._training_context import finalize_training_run, prepare_training_context


def _zero_grad(optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _adapt_targets(yb, logits, loss_type: str):
    return _adapt_targets_impl(torch, yb, logits, loss_type)


def _pred_accuracy(logits, targets, loss_type: str) -> float:
    return _pred_accuracy_impl(torch, logits, targets, loss_type)


def _eval(model, loader, criterion, device, loss_type: str = 'CrossEntropyLoss'):
    return _eval_impl(torch, model, loader, criterion, device, loss_type)


def _optimizer_params(model, optimizer_cfg: dict[str, Any]):
    """Returns param groups for the optimizer.

    Modifies `optimizer_cfg` in-place by removing weight-decay helper keys.
    """
    weight_decay = float(optimizer_cfg.get('weight_decay', 0.0) or 0.0)
    exclude_bias_norm = parse_bool(
        optimizer_cfg.pop('exclude_bias_norm_weight_decay', True),
        label='optimizer.exclude_bias_norm_weight_decay',
    )
    if weight_decay <= 0.0 or not exclude_bias_norm:
        return model.parameters()

    norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
    )
    decay = []
    no_decay = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name == 'bias' or isinstance(module, norm_types):
                no_decay.append(param)
            else:
                decay.append(param)

    optimizer_cfg.pop('weight_decay', None)
    groups = []
    if decay:
        groups.append({'params': decay, 'weight_decay': weight_decay})
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return groups or model.parameters()


def train_from_config(cfg: dict[str, Any]):
    if torch is None:
        _choose_device('auto')
    ctx = prepare_training_context(
        cfg=cfg,
        torch=torch,
        choose_device=_choose_device,
        optimizer_params_builder=_optimizer_params,
        build_model_fn=build_model,
        build_loss_fn=build_loss,
        build_optimizer_fn=build_optimizer,
        build_scheduler_fn=build_scheduler,
    )
    run_state = TrainingRunState(best_val_acc=float('-inf'))

    with (ctx.run_dir / 'metrics.jsonl').open('w', encoding='utf-8') as metrics_file:
        for epoch in range(1, ctx.epochs + 1):
            if hasattr(ctx.train_loader.dataset, 'set_epoch'):
                ctx.train_loader.dataset.set_epoch(epoch)
            epoch_t0 = time.perf_counter()
            train_metrics = run_train_epoch(
                torch,
                model=ctx.model,
                train_loader=ctx.train_loader,
                criterion=ctx.criterion,
                optimizer=ctx.optimizer,
                scaler=ctx.scaler,
                device=ctx.device,
                loss_type=ctx.loss_type,
                grad_accum_steps=ctx.grad_accum_steps,
                zero_grad=_zero_grad,
            )
            val_metrics = _eval(ctx.model, ctx.val_loader, ctx.criterion, ctx.device, ctx.loss_type)
            step_scheduler(torch, ctx.scheduler, val_metrics=val_metrics)
            epoch_time_s = time.perf_counter() - epoch_t0
            row = _build_epoch_row(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=ctx.optimizer.param_groups[0]['lr'],
                epoch_time_s=epoch_time_s,
            )
            _write_metrics_row(metrics_file, row)
            improved = handle_epoch_artifacts(
                torch,
                run_state=run_state,
                run_dir=ctx.run_dir,
                epoch=epoch,
                save_every_n_epochs=ctx.save_every_n_epochs,
                best_model_path=ctx.best_model_path,
                checkpoint_path_for_epoch=_checkpoint_path,
                model_state=ctx.model.state_dict(),
                val_acc=val_metrics['acc'],
                min_delta=ctx.min_delta,
            )
            print(
                _epoch_log_message(
                    epoch=epoch,
                    epochs=ctx.epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=ctx.optimizer.param_groups[0]['lr'],
                    epoch_time_s=epoch_time_s,
                    saved_best=improved,
                )
            )
            if should_stop_early(run_state, early_stop_patience=ctx.early_stop_patience):
                print(f'Early stopping after {epoch} epochs; best val_acc={run_state.best_val_acc * 100:.2f}%.')
                break

    test_metrics = None
    if ctx.test_loader is not None and ctx.best_model_path.exists():
        ctx.model.load_state_dict(load_best_model_state(torch, ctx.best_model_path, device=ctx.device))
        test_metrics = _eval(ctx.model, ctx.test_loader, ctx.criterion, ctx.device, ctx.loss_type)

    return finalize_training_run(
        ctx=ctx,
        run_state=run_state,
        test_metrics=test_metrics,
    )
