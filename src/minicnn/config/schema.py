from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProjectConfig:
    name: str = "minicnn"
    run_name: str = "default"
    artifacts_root: str = "artifacts"


@dataclass
class BackendConfig:
    type: str = "cuda"
    legacy_entrypoint: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 50
    eval_max_batches: int | None = None
    n_train: int = 45000
    n_val: int = 5000
    train_batch_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    dataset_seed: int = 42
    init_seed: int = 42
    train_seed: int = 42
    random_crop_padding: int = 0
    horizontal_flip: bool = True
    early_stop_patience: int = 8
    min_delta: float = 0.05
    grad_accum_steps: int = 1
    max_steps_per_epoch: int | None = None


@dataclass
class OptimConfig:
    optimizer_type: str = "sgd"   # "sgd" or "adam" for cuda_legacy backend
    lr_conv1: float = 0.005
    lr_conv: float = 0.005
    lr_fc: float = 0.005
    lr_plateau_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-5
    momentum: float = 0.9
    leaky_alpha: float = 0.1
    weight_decay: float = 5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_conv: float = 5.0
    grad_clip_fc: float = 1.0
    grad_clip_bias: float = 5.0
    grad_pool_clip: float = 1.0
    grad_clip_global: float = 0.0
    conv_grad_spatial_normalize: bool = False


@dataclass
class ModelConfig:
    c_in: int = 3          # input channels
    h: int = 32
    w: int = 32
    kh: int = 3
    kw: int = 3
    fc_out: int = 10       # number of output classes
    conv_layers: list = field(default_factory=lambda: [
        {'out_c': 32, 'pool': False},
        {'out_c': 32, 'pool': True},
        {'out_c': 64, 'pool': False},
        {'out_c': 64, 'pool': True},
    ])


@dataclass
class LossConfig:
    loss_type: str = "cross_entropy"  # "cross_entropy" | "mse" | "bce" for cuda_legacy


@dataclass
class RuntimeConfig:
    grad_debug: bool = False
    grad_debug_batches: int = 1
    best_model_filename: str = "best_model_split.npz"
    amp: bool = False
    profile: bool = False
    profile_warmup_steps: int = 3
    profile_active_steps: int = 10
    save_every_n_epochs: int = 0
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class LoggingConfig:
    console: bool = True
    jsonl: bool = True
    metrics_filename: str = "metrics.jsonl"


@dataclass
class CallbackConfig:
    checkpoint: bool = True
    summary: bool = True


@dataclass
class FrameworkConfig:
    module_system: bool = True
    optimizer_registry: bool = True
    scheduler_registry: bool = True
    healthcheck_on_info: bool = True


@dataclass
class SchedulerConfig:
    type: str = "plateau"
    enabled: bool = True
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-5


@dataclass
class ExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
