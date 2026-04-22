from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minicnn.config.parsing import parse_bool

from .builder import build_loss, build_model, build_optimizer, build_scheduler
from .data import create_dataloaders, create_test_dataloader
from .device import _choose_device, torch
from .reporting import (
    _best_model_path,
    _build_epoch_row,
    _build_training_summary,
    _checkpoint_path,
    _epoch_log_message,
    _write_metrics_row,
)
from .runtime import create_run_dir, dump_summary
from ._training_steps import (
    adapt_targets as _adapt_targets_impl,
    evaluate_model as _eval_impl,
    pred_accuracy as _pred_accuracy_impl,
    run_train_epoch,
)


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


def train_from_config(cfg: dict[str, Any]) -> Path:
    if torch is None:
        _choose_device('auto')
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    run_dir = create_run_dir(cfg)
    device = _choose_device(str(train_cfg.get('device', 'auto')))

    augmentation_cfg = cfg.get('augmentation', {})
    train_loader, val_loader = create_dataloaders(dataset_cfg, train_cfg, augmentation_cfg=augmentation_cfg)
    test_loader = create_test_dataloader(dataset_cfg, train_cfg)
    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('seed', 42)))
    torch.manual_seed(init_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(init_seed)
    model = build_model(model_cfg, input_shape=input_shape).to(device)
    criterion = build_loss(cfg.get('loss', {'type': 'CrossEntropyLoss'}))
    optimizer_cfg = dict(cfg.get('optimizer', {'type': 'SGD', 'lr': 0.01}))
    optimizer = build_optimizer(_optimizer_params(model, optimizer_cfg), optimizer_cfg)
    scheduler = build_scheduler(optimizer, cfg.get('scheduler'))
    amp_enabled = parse_bool(train_cfg.get('amp', False), label='train.amp') and device.type == 'cuda'
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    loss_type = str(cfg.get('loss', {}).get('type', 'CrossEntropyLoss'))
    best_val_acc = float('-inf')
    best_model_path = _best_model_path(run_dir)
    metrics_path = run_dir / 'metrics.jsonl'
    grad_accum_steps = max(1, int(train_cfg.get('grad_accum_steps', 1)))
    epochs = int(train_cfg.get('epochs', 1))
    early_stop_patience = int(train_cfg.get('early_stop_patience', 0) or 0)
    min_delta = float(train_cfg.get('min_delta', 0.0) or 0.0)
    epochs_no_improve = 0
    runtime_cfg = cfg.get('runtime', {})
    save_every_n_epochs = int(runtime_cfg.get('save_every_n_epochs', train_cfg.get('save_every_n_epochs', 0)) or 0)
    periodic_checkpoints: list[str] = []

    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        for epoch in range(1, epochs + 1):
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
            epoch_t0 = time.perf_counter()
            train_metrics = run_train_epoch(
                torch,
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                loss_type=loss_type,
                grad_accum_steps=grad_accum_steps,
                zero_grad=_zero_grad,
            )
            val_metrics = _eval(model, val_loader, criterion, device, loss_type)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            epoch_time_s = time.perf_counter() - epoch_t0
            row = _build_epoch_row(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]['lr'],
                epoch_time_s=epoch_time_s,
            )
            _write_metrics_row(metrics_file, row)
            if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
                checkpoint_path = _checkpoint_path(run_dir, epoch)
                torch.save({'epoch': epoch, 'model_state': model.state_dict()}, checkpoint_path)
                periodic_checkpoints.append(str(checkpoint_path))
            improved = val_metrics['acc'] > best_val_acc + min_delta
            if improved:
                best_val_acc = val_metrics['acc']
                epochs_no_improve = 0
                torch.save({'model_state': model.state_dict()}, best_model_path)
                save_msg = ' saved_best'
            else:
                epochs_no_improve += 1
            print(
                _epoch_log_message(
                    epoch=epoch,
                    epochs=epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=optimizer.param_groups[0]['lr'],
                    epoch_time_s=epoch_time_s,
                    saved_best=improved,
                )
            )
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                print(f'Early stopping after {epoch} epochs; best val_acc={best_val_acc * 100:.2f}%.')
                break

    test_metrics = None
    if test_loader is not None and best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        test_metrics = _eval(model, test_loader, criterion, device, loss_type)

    summary = _build_training_summary(
        device=device,
        run_dir=run_dir,
        best_model_path=best_model_path,
        input_shape=input_shape,
        model_cfg=model_cfg,
        cfg=cfg,
        periodic_checkpoints=periodic_checkpoints,
        test_metrics=test_metrics,
    )
    dump_summary(run_dir, summary)
    return run_dir
