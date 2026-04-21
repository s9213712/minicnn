# Model Artifacts

MiniCNN does **not** use one universal model file format across all backends.

That is intentional today: each backend saves the checkpoint format that matches
its runtime boundary. The tradeoff is that reuse is backend-local unless a
conversion tool is added later.

## Artifact Matrix

| Backend / path | Typical file | Format | What is stored | Reuse target |
|---|---|---|---|---|
| `train-flex` / `train-dual engine.backend=torch` | `*_best.pt` | PyTorch checkpoint | `{'model_state': state_dict}` | torch only |
| periodic torch checkpoint | `*_epoch_3.pt` | PyTorch checkpoint | `{'epoch': ..., 'model_state': ...}` | torch only |
| `train-autograd` | `*_autograd_best.npz` | NumPy archive | `model.state_dict()` arrays | autograd only |
| `train-native` / `engine.backend=cuda_native` | `*_best.npz` | NumPy archive | flat parameter dict such as `_w_*`, `_b_*`, running stats | cuda_native only |
| `train-dual engine.backend=cuda_legacy` | `*.npz` | NumPy archive | handcrafted CUDA checkpoint schema (`w_conv*`, `fc_*`, lr state, BN stats, epoch, val_acc) | cuda_legacy only |

`summary.json` is the common pointer across all of them. It records
`best_model_path`, but that path does **not** imply a unified checkpoint
schema.

## Where Files Go

- run metadata: `artifacts/<run-name-timestamp>/`
- best models: `artifacts/models/`
- per-run config snapshot: `config.yaml`
- per-run metrics: `metrics.jsonl`
- per-run summary: `summary.json`

## Inspect A Checkpoint

Use the CLI:

```bash
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn inspect-checkpoint --path artifacts/models/example_autograd_best.npz
```

What it does:

- `.npz`: lists keys, shapes, dtypes, and a guessed checkpoint kind
- `.pt` / `.pth`: lists top-level keys and `model_state` keys

Notes:

- inspecting `.pt` / `.pth` still requires PyTorch
- inspecting `.npz` does not

## Export To A Generic Torch Checkpoint

For supported sources, MiniCNN can export to a standard PyTorch checkpoint:

```bash
minicnn export-torch-checkpoint \
  --path artifacts/models/my_run_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/my_run_autograd_export.pt
```

Currently supported:

- `autograd` `.npz` -> torch `.pt`
- `cuda_native` `.npz` -> torch `.pt`

Currently not supported:

- `cuda_legacy` `.npz` -> torch `.pt`

Why `cuda_legacy` is excluded:

- its checkpoint schema is tied to handcrafted runtime geometry
- it stores runtime-specific training state, not a frontend-level portable model format
- a safe export path would need an explicit geometry-to-module conversion layer

## Reuse Examples

### Torch / flex

```python
import torch

from minicnn.flex.builder import build_model
from minicnn.flex.config import load_flex_config

cfg = load_flex_config('configs/flex_cnn.yaml')
model = build_model(cfg['model'], input_shape=cfg['dataset']['input_shape'])
ckpt = torch.load('artifacts/models/my_run_best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state'])
model.eval()
```

### Autograd

```python
import numpy as np

from minicnn.models.builder import build_model_from_config

cfg = {
    'layers': [
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
}
model = build_model_from_config(cfg, input_shape=(1, 28, 28))
ckpt = np.load('artifacts/models/my_run_autograd_best.npz')
model.load_state_dict({k: ckpt[k] for k in ckpt.files})
model.eval()
```

Or export it once and reuse it as a normal torch checkpoint:

```bash
minicnn export-torch-checkpoint \
  --path artifacts/models/my_run_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/my_run_autograd_export.pt
```

### cuda_native

`cuda_native` best files are flat NumPy parameter dicts. Reuse means rebuilding
the same graph, then passing the loaded params back into the executor:

```python
import numpy as np

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.executor import ForwardExecutor

graph = build_cuda_native_graph(model_cfg, input_shape=[1, 3, 32, 32])
params_file = np.load('artifacts/models/my_run_best.npz')
params = {k: params_file[k] for k in params_file.files}
executor = ForwardExecutor()
logits = executor.run_inference(graph, x_batch, params=params, mode='eval')
```

If the config maps cleanly onto the torch/flex model builder, you can also export
it to torch:

```bash
minicnn export-torch-checkpoint \
  --path artifacts/models/my_run_best.npz \
  --config configs/dual_backend_cnn.yaml \
  --output artifacts/models/my_run_native_export.pt
```

### cuda_legacy

`cuda_legacy` checkpoints are **not** a generic state dict. They are tied to
the handcrafted CUDA geometry and are reloaded through
`reload_weights_from_checkpoint()` in
`src/minicnn/training/checkpoints.py`.

If you want to reuse them, rebuild the same `CudaNetGeometry` first, then use
that reload path. They are not drop-in compatible with torch, autograd, or
cuda_native.

## Important Boundaries

- `torch` `.pt` files are not directly loadable into autograd or cuda_native.
- `autograd` and `cuda_native` both use `.npz`, but the schemas are different.
- `cuda_legacy` `.npz` is a handcrafted runtime checkpoint, not a frontend-level model exchange format.
- if you need cross-backend reuse, add an explicit conversion layer instead of assuming filename parity means compatibility.
- exported torch checkpoints are only as portable as the config they were built from; use the same architecture config when exporting.

---

# 模型產物

MiniCNN **沒有**在所有 backend 間使用單一統一的模型檔格式。

這是目前刻意的設計：每個 backend 儲存最符合自己 runtime 邊界的
checkpoint。代價是模型復用目前主要侷限在同 backend 內，除非之後再加轉換工具。

## 產物矩陣

| Backend / 路徑 | 常見檔名 | 格式 | 儲存內容 | 可復用目標 |
|---|---|---|---|---|
| `train-flex` / `train-dual engine.backend=torch` | `*_best.pt` | PyTorch checkpoint | `{'model_state': state_dict}` | 僅 torch |
| torch 週期 checkpoint | `*_epoch_3.pt` | PyTorch checkpoint | `{'epoch': ..., 'model_state': ...}` | 僅 torch |
| `train-autograd` | `*_autograd_best.npz` | NumPy archive | `model.state_dict()` 陣列 | 僅 autograd |
| `train-native` / `engine.backend=cuda_native` | `*_best.npz` | NumPy archive | `_w_*`、`_b_*`、running stats 等平坦參數 dict | 僅 cuda_native |
| `train-dual engine.backend=cuda_legacy` | `*.npz` | NumPy archive | 手寫 CUDA checkpoint schema（`w_conv*`、`fc_*`、lr state、BN stats、epoch、val_acc） | 僅 cuda_legacy |

`summary.json` 是這些產物共用的索引點。它會記錄 `best_model_path`，但這個
path **不代表** checkpoint schema 已統一。

## 檔案位置

- run metadata：`artifacts/<run-name-timestamp>/`
- best models：`artifacts/models/`
- 每次 run 的 config 快照：`config.yaml`
- 每次 run 的 metrics：`metrics.jsonl`
- 每次 run 的 summary：`summary.json`

## 檢查 checkpoint

可直接用 CLI：

```bash
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn inspect-checkpoint --path artifacts/models/example_autograd_best.npz
```

它會做的事：

- `.npz`：列出 keys、shape、dtype，以及推測的 checkpoint kind
- `.pt` / `.pth`：列出 top-level keys 與 `model_state` keys

注意：

- 檢查 `.pt` / `.pth` 仍需要 PyTorch
- 檢查 `.npz` 不需要

## 匯出成通用 torch checkpoint

對支援的來源格式，MiniCNN 現在可以匯出成標準 PyTorch checkpoint：

```bash
minicnn export-torch-checkpoint \
  --path artifacts/models/my_run_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/my_run_autograd_export.pt
```

目前支援：

- `autograd` `.npz` -> torch `.pt`
- `cuda_native` `.npz` -> torch `.pt`

目前不支援：

- `cuda_legacy` `.npz` -> torch `.pt`

`cuda_legacy` 先不支援的原因：

- 它的 checkpoint schema 綁定手寫 runtime geometry
- 裡面存的是 runtime-specific training state，不是前端層可攜模型格式
- 若要安全轉換，需額外實作 geometry-to-module conversion layer

## 復用示範

### Torch / flex

```python
import torch

from minicnn.flex.builder import build_model
from minicnn.flex.config import load_flex_config

cfg = load_flex_config('configs/flex_cnn.yaml')
model = build_model(cfg['model'], input_shape=cfg['dataset']['input_shape'])
ckpt = torch.load('artifacts/models/my_run_best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state'])
model.eval()
```

### Autograd

```python
import numpy as np

from minicnn.models.builder import build_model_from_config

cfg = {
    'layers': [
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
}
model = build_model_from_config(cfg, input_shape=(1, 28, 28))
ckpt = np.load('artifacts/models/my_run_autograd_best.npz')
model.load_state_dict({k: ckpt[k] for k in ckpt.files})
model.eval()
```

如果你想把它先轉成一般 torch checkpoint 再交給外部使用：

```bash
minicnn export-torch-checkpoint \
  --path artifacts/models/my_run_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/my_run_autograd_export.pt
```

### cuda_native

`cuda_native` best 檔是平坦的 NumPy parameter dict。要復用時，需先重建同一張
graph，再把載入後的 params 傳回 executor：

```python
import numpy as np

from minicnn.cuda_native.api import build_cuda_native_graph
from minicnn.cuda_native.executor import ForwardExecutor

graph = build_cuda_native_graph(model_cfg, input_shape=[1, 3, 32, 32])
params_file = np.load('artifacts/models/my_run_best.npz')
params = {k: params_file[k] for k in params_file.files}
executor = ForwardExecutor()
logits = executor.run_inference(graph, x_batch, params=params, mode='eval')
```

若這份 config 能乾淨映射到 torch/flex model builder，也可以匯出成 torch：

```bash
minicnn export-torch-checkpoint \
  --path artifacts/models/my_run_best.npz \
  --config configs/dual_backend_cnn.yaml \
  --output artifacts/models/my_run_native_export.pt
```

### cuda_legacy

`cuda_legacy` checkpoint **不是** generic state dict。它綁定手寫 CUDA 的幾何與
權重配置，需透過 `src/minicnn/training/checkpoints.py` 裡的
`reload_weights_from_checkpoint()` 走回載入流程。

若要復用，先重建相同的 `CudaNetGeometry`，再走那條 reload 路徑。它不能直接拿
去餵 torch、autograd 或 cuda_native。

## 重要邊界

- `torch` `.pt` 檔不能直接載入 autograd 或 cuda_native。
- `autograd` 和 `cuda_native` 都用 `.npz`，但 schema 不同。
- `cuda_legacy` `.npz` 是手寫 runtime checkpoint，不是前端層的模型交換格式。
- 若你需要跨 backend 復用，請加明確的 conversion layer，不要假設檔名相似就代表相容。
- 匯出的 torch checkpoint 能否被外部安全使用，仍取決於匯出時使用的架構 config 是否一致。
