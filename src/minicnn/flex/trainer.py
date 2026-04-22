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


def _zero_grad(optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _adapt_targets(yb, logits, loss_type: str):
    """Convert integer class labels to the shape/dtype expected by the loss function."""
    if loss_type == 'CrossEntropyLoss':
        return yb  # integer labels are correct
    n, out = logits.shape[0], logits.shape[1] if logits.dim() > 1 else 1
    if loss_type == 'MSELoss':
        dense = torch.zeros(n, out, dtype=torch.float32, device=yb.device)
        dense[torch.arange(n, device=yb.device), yb.long()] = 1.0
        return dense
    if loss_type == 'BCEWithLogitsLoss':
        if out != 1:
            raise ValueError(
                f'BCEWithLogitsLoss with flex trainer requires out_features=1 (binary classification), '
                f'but the model outputs {out} logits. '
                'Use CrossEntropyLoss for multi-class classification.'
            )
        yb_int = yb.long()
        if int(yb_int.min().item()) < 0 or int(yb_int.max().item()) > 1:
            bad_min, bad_max = int(yb_int.min().item()), int(yb_int.max().item())
            raise ValueError(
                f'BCEWithLogitsLoss binary classification labels must be in {{0, 1}}, '
                f'but got min={bad_min}, max={bad_max}. '
                'Use CrossEntropyLoss for multi-class classification.'
            )
        return yb.float().reshape(n, 1)
    return yb


def _pred_accuracy(logits, targets, loss_type: str) -> float:
    """Compute prediction accuracy appropriate for the loss function."""
    if loss_type == 'BCEWithLogitsLoss':
        preds = (logits.squeeze(1) >= 0.0).long()
        return float((preds == targets.long()).float().mean().item())
    preds = logits.argmax(dim=1)
    return float((preds == targets.long()).float().mean().item())


def _eval(model, loader, criterion, device, loss_type: str = 'CrossEntropyLoss'):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            adapted = _adapt_targets(yb, logits, loss_type)
            loss = criterion(logits, adapted)
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            acc_sum += _pred_accuracy(logits, yb, loss_type) * bs
            count += bs
    return {'loss': loss_sum / max(count, 1), 'acc': acc_sum / max(count, 1)}


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
            model.train()
            _zero_grad(optimizer)
            running_loss = 0.0
            running_acc = 0.0
            seen = 0
            n_batches = len(train_loader)
            for step, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(device)
                yb = yb.to(device)
                if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                    autocast_ctx = torch.amp.autocast('cuda', enabled=scaler.is_enabled())
                else:  # pragma: no cover
                    autocast_ctx = torch.cuda.amp.autocast(enabled=scaler.is_enabled())
                with autocast_ctx:
                    logits = model(xb)
                    adapted_yb = _adapt_targets(yb, logits, loss_type)
                    loss = criterion(logits, adapted_yb) / grad_accum_steps
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0 or step == n_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    _zero_grad(optimizer)
                bs = xb.shape[0]
                running_loss += float(loss.item()) * grad_accum_steps * bs
                running_acc += _pred_accuracy(logits, yb, loss_type) * bs
                seen += bs

            train_metrics = {'loss': running_loss / max(seen, 1), 'acc': running_acc / max(seen, 1)}
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
