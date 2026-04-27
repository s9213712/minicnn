from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minicnn.config.parsing import parse_bool
from minicnn.random import set_global_seed

from .data import create_dataloaders, create_test_dataloader
from .reporting import _best_model_path, _build_training_summary
from .runtime import create_run_dir, dump_summary


@dataclass
class FlexTrainingContext:
    cfg: dict[str, Any]
    dataset_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    run_dir: Path
    device: Any
    train_loader: Any
    val_loader: Any
    test_loader: Any
    input_shape: tuple[int, ...]
    model: Any
    criterion: Any
    optimizer: Any
    scheduler: Any
    scaler: Any
    loss_type: str
    best_model_path: Path
    grad_accum_steps: int
    epochs: int
    early_stop_patience: int
    min_delta: float
    save_every_n_epochs: int


def prepare_training_context(
    *,
    cfg: dict[str, Any],
    torch,
    choose_device,
    optimizer_params_builder,
    build_model_fn,
    build_loss_fn,
    build_optimizer_fn,
    build_scheduler_fn,
) -> FlexTrainingContext:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})
    device = choose_device(str(train_cfg.get('device', 'auto')))

    augmentation_cfg = cfg.get('augmentation', {})
    train_loader, val_loader = create_dataloaders(dataset_cfg, train_cfg, augmentation_cfg=augmentation_cfg)
    test_loader = create_test_dataloader(dataset_cfg, train_cfg)
    input_shape = tuple(dataset_cfg.get('input_shape', [3, 32, 32]))
    init_seed = int(train_cfg.get('init_seed', dataset_cfg.get('dataset_seed', dataset_cfg.get('seed', 42))))
    set_global_seed(init_seed)
    torch.manual_seed(init_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(init_seed)
    model = build_model_fn(model_cfg, input_shape=input_shape).to(device)
    train_seed = int(train_cfg.get('train_seed', train_cfg.get('seed', dataset_cfg.get('seed', 42))))
    set_global_seed(train_seed)
    torch.manual_seed(train_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(train_seed)
    criterion = build_loss_fn(cfg.get('loss', {'type': 'CrossEntropyLoss'}))
    optimizer_cfg = dict(cfg.get('optimizer', {'type': 'SGD', 'lr': 0.01}))
    optimizer = build_optimizer_fn(optimizer_params_builder(model, optimizer_cfg), optimizer_cfg)
    scheduler = build_scheduler_fn(optimizer, cfg.get('scheduler'))
    amp_enabled = parse_bool(train_cfg.get('amp', False), label='train.amp') and device.type == 'cuda'
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    runtime_cfg = cfg.get('runtime', {})
    run_dir = create_run_dir(cfg)
    return FlexTrainingContext(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        run_dir=run_dir,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=input_shape,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_type=str(cfg.get('loss', {}).get('type', 'CrossEntropyLoss')),
        best_model_path=_best_model_path(run_dir),
        grad_accum_steps=max(1, int(train_cfg.get('grad_accum_steps', 1))),
        epochs=int(train_cfg.get('epochs', 1)),
        early_stop_patience=int(train_cfg.get('early_stop_patience', 0) or 0),
        min_delta=float(train_cfg.get('min_delta', 0.0) or 0.0),
        save_every_n_epochs=int(runtime_cfg.get('save_every_n_epochs', train_cfg.get('save_every_n_epochs', 0)) or 0),
    )


def finalize_training_run(
    *,
    ctx: FlexTrainingContext,
    run_state,
    test_metrics,
) -> Path:
    summary = _build_training_summary(
        device=ctx.device,
        run_dir=ctx.run_dir,
        best_model_path=ctx.best_model_path,
        input_shape=ctx.input_shape,
        model_cfg=ctx.model_cfg,
        cfg=ctx.cfg,
        periodic_checkpoints=run_state.periodic_checkpoints,
        test_metrics=test_metrics,
    )
    dump_summary(ctx.run_dir, summary)
    return ctx.run_dir
