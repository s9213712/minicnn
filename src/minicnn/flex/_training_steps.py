from __future__ import annotations

from typing import Any


def adapt_targets(torch, yb, logits, loss_type: str):
    """Convert integer class labels to the shape/dtype expected by the loss function."""
    if loss_type == 'CrossEntropyLoss':
        return yb
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


def pred_accuracy(torch, logits, targets, loss_type: str) -> float:
    """Compute prediction accuracy appropriate for the loss function."""
    if loss_type == 'BCEWithLogitsLoss':
        preds = (logits.squeeze(1) >= 0.0).long()
        return float((preds == targets.long()).float().mean().item())
    preds = logits.argmax(dim=1)
    return float((preds == targets.long()).float().mean().item())


def evaluate_model(torch, model, loader, criterion, device, loss_type: str = 'CrossEntropyLoss'):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            adapted = adapt_targets(torch, yb, logits, loss_type)
            loss = criterion(logits, adapted)
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            acc_sum += pred_accuracy(torch, logits, yb, loss_type) * bs
            count += bs
    return {'loss': loss_sum / max(count, 1), 'acc': acc_sum / max(count, 1)}


def run_train_epoch(
    torch,
    *,
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    loss_type: str,
    grad_accum_steps: int,
    zero_grad,
) -> dict[str, float]:
    model.train()
    zero_grad(optimizer)
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
            adapted_yb = adapt_targets(torch, yb, logits, loss_type)
            loss = criterion(logits, adapted_yb) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == n_batches:
            scaler.step(optimizer)
            scaler.update()
            zero_grad(optimizer)
        bs = xb.shape[0]
        running_loss += float(loss.item()) * grad_accum_steps * bs
        running_acc += pred_accuracy(torch, logits, yb, loss_type) * bs
        seen += bs

    return {
        'loss': running_loss / max(seen, 1),
        'acc': running_acc / max(seen, 1),
    }
