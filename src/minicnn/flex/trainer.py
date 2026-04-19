from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .builder import build_loss, build_model, build_optimizer, build_scheduler
from .data import create_dataloaders, create_test_dataloader
from .runtime import create_run_dir, dump_summary
from minicnn.paths import BEST_MODELS_ROOT

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _choose_device(device_cfg: str):
    if device_cfg == 'cpu':
        return torch.device('cpu')
    if device_cfg == 'cuda':
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def _eval(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            acc_sum += _accuracy(logits, yb) * bs
            count += bs
    return {'loss': loss_sum / max(count, 1), 'acc': acc_sum / max(count, 1)}


def _best_model_path(run_dir: Path) -> Path:
    BEST_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return BEST_MODELS_ROOT / f'{run_dir.name}_best.pt'


def train_from_config(cfg: dict[str, Any]) -> Path:
    if torch is None:
        raise RuntimeError('PyTorch is required for train-flex')
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    run_dir = create_run_dir(cfg)
    device = _choose_device(str(train_cfg.get('device', 'auto')))

    train_loader, val_loader = create_dataloaders(dataset_cfg, train_cfg)
    test_loader = create_test_dataloader(dataset_cfg, train_cfg)
    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    model = build_model(model_cfg, input_shape=input_shape).to(device)
    criterion = build_loss(cfg.get('loss', {'type': 'CrossEntropyLoss'}))
    optimizer = build_optimizer(model.parameters(), cfg.get('optimizer', {'type': 'SGD', 'lr': 0.01}))
    scheduler = build_scheduler(optimizer, cfg.get('scheduler'))
    amp_enabled = bool(train_cfg.get('amp', False) and device.type == 'cuda')
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_acc = float('-inf')
    best_model_path = _best_model_path(run_dir)
    metrics_path = run_dir / 'metrics.jsonl'
    grad_accum_steps = max(1, int(train_cfg.get('grad_accum_steps', 1)))
    epochs = int(train_cfg.get('epochs', 1))

    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        for epoch in range(1, epochs + 1):
            epoch_t0 = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
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
                    loss = criterion(logits, yb) / grad_accum_steps
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0 or step == n_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                bs = xb.shape[0]
                running_loss += float(loss.item()) * grad_accum_steps * bs
                running_acc += _accuracy(logits, yb) * bs
                seen += bs

            train_metrics = {'loss': running_loss / max(seen, 1), 'acc': running_acc / max(seen, 1)}
            val_metrics = _eval(model, val_loader, criterion, device)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            epoch_time_s = time.perf_counter() - epoch_t0
            row = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['acc'],
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time_s': epoch_time_s,
            }
            metrics_file.write(json.dumps(row) + '\n')
            metrics_file.flush()
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                torch.save({'model_state': model.state_dict(), 'config': cfg}, best_model_path)

    test_metrics = None
    if test_loader is not None and best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        test_metrics = _eval(model, test_loader, criterion, device)

    summary = {
        'device': str(device),
        'run_dir': str(run_dir),
        'best_model_path': str(best_model_path),
        'input_shape': list(input_shape),
        'model_layers': [layer.get('type') for layer in model_cfg.get('layers', [])],
        'optimizer': cfg.get('optimizer', {}).get('type'),
        'loss': cfg.get('loss', {}).get('type'),
        'scheduler': cfg.get('scheduler', {}).get('type') if cfg.get('scheduler', {}).get('enabled') else None,
    }
    if test_metrics is not None:
        summary['test_loss'] = test_metrics['loss']
        summary['test_acc'] = test_metrics['acc']
    dump_summary(run_dir, summary)
    return run_dir
