# MiniCNN Documentation Guide

This page is the documentation entrypoint for the current repo.

If you only read one documentation file outside the README, make it this one.

## Start Here

Read these in order if you want the current, operational picture of the repo:

1. [docs/architecture.md](docs/architecture.md) — overall structure, execution paths, and module boundaries
2. [docs/backend_capabilities.md](docs/backend_capabilities.md) — what each backend really supports
3. [docs/dual_backend_guide.md](docs/dual_backend_guide.md) — how one shared config maps into the reference, native, oracle, and maintenance roles

## By Task

### I want to install MiniCNN and check that it works

- Run `minicnn smoke`
- See [README.md](README.md) for quick start
- See [docs/architecture.md](docs/architecture.md) and [docs/backend_capabilities.md](docs/backend_capabilities.md) if the result is unclear

### I want to train a model right now

- [docs/backend_capabilities.md](docs/backend_capabilities.md) — choose the right backend first
- [docs/dual_backend_guide.md](docs/dual_backend_guide.md) — shared-config path and backend routing
- [docs/model_artifacts.md](docs/model_artifacts.md) — checkpoint formats and reuse boundaries
- [templates/README.md](templates/README.md) — ready-to-edit config templates

### I want to work on the handcrafted CUDA path

- [docs/guide_project_structure.md](docs/guide_project_structure.md) — repo map of the native and Python sides
- [docs/guide_native_build.md](docs/guide_native_build.md) — build the shared library
- [docs/guide_c_api.md](docs/guide_c_api.md) — exported C API reference
- [docs/guide_layout_debug.md](docs/guide_layout_debug.md) — layout rules and debugging workflow
- [docs/guide_windows_build.md](docs/guide_windows_build.md) — manually validated Windows build guide (not CI-covered)
- [docs/cuda_batchnorm2d_evaluation.md](docs/cuda_batchnorm2d_evaluation.md) — focused note on one unresolved `cuda_legacy` maintenance area

### I want Python `ctypes` or C++ embedding examples

- [docs/guide_ctypes_example.md](docs/guide_ctypes_example.md) — Python `ctypes` example against the native library
- [docs/guide_cpp_linking.md](docs/guide_cpp_linking.md) — C++ linking path for the secondary C++ API

### I want to work on autograd or the broader frontend

- [docs/guide_autograd.md](docs/guide_autograd.md) — NumPy autograd stack as the internal correctness oracle
- [docs/guide_feature_expansion.md](docs/guide_feature_expansion.md) — wider feature-surface notes
- [docs/custom_components.md](docs/custom_components.md) — dotted-path component and dataset extension points
- [docs/generalization_roadmap.md](docs/generalization_roadmap.md) — how frontend breadth should relate to backend honesty

### I want to work on `cuda_native`

- [docs/cuda_native.md](docs/cuda_native.md) — full guide to the primary native backend direction
- [docs/cuda_native_phase5_rfc.md](docs/cuda_native_phase5_rfc.md) — future extension RFCs
- [docs/backend_capabilities.md](docs/backend_capabilities.md) — current validated support boundary

## CLI Surface

### Stable commands

```bash
minicnn build --legacy-make --check
minicnn smoke
minicnn prepare-data
minicnn train-flex --config configs/flex_cnn.yaml
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
minicnn train-autograd --config configs/autograd_tiny.yaml
minicnn compare --config configs/dual_backend_cnn.yaml
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
minicnn show-model --config configs/flex_cnn.yaml
minicnn show-graph --config configs/flex_cnn.yaml --format json
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn export-torch-checkpoint --path artifacts/models/example_autograd_best.npz --config configs/autograd_tiny.yaml --output artifacts/models/example_autograd_export.pt
minicnn compile --config configs/autograd_tiny.yaml
```

### Experimental but public commands

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
minicnn cuda-native-capabilities
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

### Inspection commands

```bash
minicnn info
minicnn smoke
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
```

`minicnn smoke` is the fastest first-run check. It confirms that:

- the repo layout is intact
- built-in configs parse
- the compiler can trace a default flex model
- both `cuda_legacy` and `cuda_native` validators still accept their reference configs

`show-model` is the frontend/config view.
`show-graph` is the primitive compiler-IR view.

## Minimum Dependency Matrix

| Command / feature | PyTorch | native `.so` | CIFAR-10 data |
|---|---:|---:|---:|
| `minicnn --help` | no | no | no |
| `minicnn validate-dual-config` | no | no | no |
| `minicnn show-cuda-mapping` | no | no | no |
| `minicnn show-model` | no | no | no |
| `minicnn show-graph` | no | no | no |
| `minicnn compile` | no | no | no |
| `minicnn train-flex` | yes | no | depends on dataset |
| `minicnn train-dual engine.backend=torch` | yes | no | depends on dataset |
| `minicnn train-dual engine.backend=cuda_legacy` | no | yes | yes |
| `minicnn train-autograd` | no | no | depends on dataset |
| `minicnn train-native` | no | no | depends on dataset |

Torch-only commands fail with a short CLI dependency message instead of an
import-time traceback.

Config and override mistakes also fail with a short message and exit code `2`
instead of a Python traceback. `healthcheck`, `doctor`, `smoke`,
`validate-*`, `show-cuda-mapping`, `show-model`, `show-graph`, and `inspect-checkpoint` emit JSON-friendly
output. If your PyTorch runtime has no CUDA support, `train.device=cuda` fails
early and tells you to switch to `auto` or `cpu`.

These diagnostic, validation, and inspection commands also accept `--format
text` for human-readable terminal output while keeping `json` as the default
machine-readable format.

Built-in config paths such as `configs/flex_cnn.yaml` and
`configs/dual_backend_cnn.yaml` fall back to project-root-relative resolution,
so they still work when `minicnn` is launched from outside the repo root.
This is still a repo-first convenience model, not a full packaged-resource
distribution.
If a built-in config still cannot be found, the CLI now tells you to pass an
explicit `--config` path.

## Document Roles

### Current operational docs

These describe the repo as it works today:

- [docs/architecture.md](docs/architecture.md)
- [docs/backend_capabilities.md](docs/backend_capabilities.md)
- [docs/dual_backend_guide.md](docs/dual_backend_guide.md)
- [docs/cuda_native.md](docs/cuda_native.md)
- [docs/custom_components.md](docs/custom_components.md)
- [docs/model_artifacts.md](docs/model_artifacts.md)
- [docs/guide_project_structure.md](docs/guide_project_structure.md)
- [docs/guide_autograd.md](docs/guide_autograd.md)
- [docs/guide_feature_expansion.md](docs/guide_feature_expansion.md)

### Focused technical notes

These are narrower deep dives, still useful, but not the best first read:

- [docs/guide_layout_debug.md](docs/guide_layout_debug.md)
- [docs/cuda_batchnorm2d_evaluation.md](docs/cuda_batchnorm2d_evaluation.md)
- [docs/cuda_native_phase5_rfc.md](docs/cuda_native_phase5_rfc.md)

### Historical or reporting documents

These are reference material, not primary onboarding docs:

- [docs/comparison_report.md](docs/comparison_report.md)
- [docs/comparison_completion_report.md](docs/comparison_completion_report.md)
- [docs/optimization_progress.md](docs/optimization_progress.md)
- [docs/benchmark_report_template.md](docs/benchmark_report_template.md)
- [docs/agent_completion_report.md](docs/agent_completion_report.md)

## Quick Navigation

If you are unsure where to go next:

- Need the truth about support boundaries: [docs/backend_capabilities.md](docs/backend_capabilities.md)
- Need to understand one shared config across backends: [docs/dual_backend_guide.md](docs/dual_backend_guide.md)
- Need native build/debug details: [docs/guide_project_structure.md](docs/guide_project_structure.md) and [docs/guide_layout_debug.md](docs/guide_layout_debug.md)
- Need extension points: [docs/custom_components.md](docs/custom_components.md)
- Need checkpoint format and model reuse details: [docs/model_artifacts.md](docs/model_artifacts.md)
- Need experimental native graph backend context: [docs/cuda_native.md](docs/cuda_native.md)

## Rule Of Thumb

- Use `torch/flex` as the reference implementation and first stop for new features.
- Use `train-autograd` when you need a CPU-side correctness oracle or framework-level learning path.
- Use `cuda_native` when you are pushing the native backend forward; it is the main native growth path, but still experimental.
- Use `cuda_legacy` only inside its validator boundary and treat it as maintenance-only historical code.

---

# MiniCNN 文件導覽

這份文件是目前 repo 的文件入口頁。

如果 README 之外你只讀一份文件，就先讀這份。

## 從哪裡開始

如果你想先掌握目前 repo 的實際運作狀態，建議依序閱讀：

1. [docs/architecture.md](docs/architecture.md) — 整體結構、執行路徑與模組邊界
2. [docs/backend_capabilities.md](docs/backend_capabilities.md) — 各 backend 真正支援的能力
3. [docs/dual_backend_guide.md](docs/dual_backend_guide.md) — 同一份 shared config 如何映射到 reference、native、oracle、maintenance 這些角色

## 依任務找文件

### 我想先安裝並確認 MiniCNN 能跑

- 先執行 `minicnn smoke`
- Quick start 見 [README.md](README.md)
- 若結果看不懂，再看 [docs/architecture.md](docs/architecture.md) 和 [docs/backend_capabilities.md](docs/backend_capabilities.md)

### 我現在就想訓練模型

- [docs/backend_capabilities.md](docs/backend_capabilities.md) — 先選對 backend
- [docs/dual_backend_guide.md](docs/dual_backend_guide.md) — shared-config 路徑與 backend routing
- [docs/model_artifacts.md](docs/model_artifacts.md) — checkpoint 格式與復用邊界
- [templates/README.md](templates/README.md) — 可直接修改的 config templates

### 我想處理手寫 CUDA 路徑

- [docs/guide_project_structure.md](docs/guide_project_structure.md) — native 與 Python 兩側的 repo 結構圖
- [docs/guide_native_build.md](docs/guide_native_build.md) — shared library 編譯流程
- [docs/guide_c_api.md](docs/guide_c_api.md) — 匯出的 C API 參考
- [docs/guide_layout_debug.md](docs/guide_layout_debug.md) — layout 規則與 debug 流程
- [docs/guide_windows_build.md](docs/guide_windows_build.md) — 已手動驗證的 Windows 建置指南（尚未納入 CI）
- [docs/cuda_batchnorm2d_evaluation.md](docs/cuda_batchnorm2d_evaluation.md) — `cuda_legacy` 尚未完成的維護型 BatchNorm2d 評估

### 我想看 Python `ctypes` 或 C++ embedding 範例

- [docs/guide_ctypes_example.md](docs/guide_ctypes_example.md) — 使用 Python `ctypes` 呼叫 native library 的範例
- [docs/guide_cpp_linking.md](docs/guide_cpp_linking.md) — Secondary C++ API 的連結方式

### 我想處理 autograd 或更廣的 frontend

- [docs/guide_autograd.md](docs/guide_autograd.md) — 作為內部 correctness oracle 的 NumPy autograd stack
- [docs/guide_feature_expansion.md](docs/guide_feature_expansion.md) — 更廣的 feature surface 說明
- [docs/custom_components.md](docs/custom_components.md) — dotted-path component / dataset extension points
- [docs/generalization_roadmap.md](docs/generalization_roadmap.md) — frontend 廣度該如何和 backend 邊界保持誠實

### 我想處理 `cuda_native`

- [docs/cuda_native.md](docs/cuda_native.md) — 主要 native backend 方向的完整指南
- [docs/cuda_native_phase5_rfc.md](docs/cuda_native_phase5_rfc.md) — 後續擴充 RFC
- [docs/backend_capabilities.md](docs/backend_capabilities.md) — 目前已驗證的支援邊界

## CLI 介面

### 穩定指令

```bash
minicnn build --legacy-make --check
minicnn smoke
minicnn prepare-data
minicnn train-flex --config configs/flex_cnn.yaml
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
minicnn train-autograd --config configs/autograd_tiny.yaml
minicnn compare --config configs/dual_backend_cnn.yaml
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn export-torch-checkpoint --path artifacts/models/example_autograd_best.npz --config configs/autograd_tiny.yaml --output artifacts/models/example_autograd_export.pt
minicnn compile --config configs/autograd_tiny.yaml
```

### 實驗性但公開的指令

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
minicnn cuda-native-capabilities
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

### 檢查類指令

```bash
minicnn info
minicnn smoke
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
```

`minicnn smoke` 是最快的 first-run 自檢。它會確認：

- repo 結構完整
- 內建 config 能成功解析
- compiler 能 trace 預設 flex 模型
- `cuda_legacy` 與 `cuda_native` validator 仍接受各自的參考 config

## 最小依賴矩陣

| 指令 / 功能 | 需要 PyTorch | 需要 native `.so` | 需要 CIFAR-10 data |
|---|---:|---:|---:|
| `minicnn --help` | 否 | 否 | 否 |
| `minicnn validate-dual-config` | 否 | 否 | 否 |
| `minicnn show-cuda-mapping` | 否 | 否 | 否 |
| `minicnn show-model` | 否 | 否 | 否 |
| `minicnn show-graph` | 否 | 否 | 否 |
| `minicnn compile` | 否 | 否 | 否 |
| `minicnn train-flex` | 是 | 否 | 視 dataset 而定 |
| `minicnn train-dual engine.backend=torch` | 是 | 否 | 視 dataset 而定 |
| `minicnn train-dual engine.backend=cuda_legacy` | 否 | 是 | 是 |
| `minicnn train-autograd` | 否 | 否 | 視 dataset 而定 |
| `minicnn train-native` | 否 | 否 | 視 dataset 而定 |

torch-only 指令現在會輸出簡短的 CLI 依賴訊息，不再在 import 階段直接丟
traceback。

config 或 override 寫錯時，也會以簡短訊息和 exit code `2` 失敗，而不是吐出
Python traceback。`healthcheck`、`doctor`、`smoke`、`validate-*`、
`show-cuda-mapping`、`show-model`、`show-graph`、`inspect-checkpoint` 都會輸出 JSON-friendly
結果；若目前 PyTorch runtime 不支援 CUDA，`train.device=cuda`
也會提早失敗並提示改用 `auto` 或 `cpu`。

這些診斷、檢查與驗證命令也接受 `--format text`，方便直接在終端查看；`json`
仍是預設的機器可讀格式。

像 `configs/flex_cnn.yaml`、`configs/dual_backend_cnn.yaml` 這類內建 config
路徑，必要時會自動以 project root 為基準解析，所以不必強制在 repo root
下執行 `minicnn`。
但這仍是 repo-first 的便利模型，不是完整 packaged-resource 發行方式。
如果內建 config 仍然找不到，CLI 現在會明確提示改傳顯式 `--config` 路徑。

## 文件角色

### 目前仍屬主線操作文件

這些文件描述的是 repo 今天實際怎麼運作：

- [docs/architecture.md](docs/architecture.md)
- [docs/backend_capabilities.md](docs/backend_capabilities.md)
- [docs/dual_backend_guide.md](docs/dual_backend_guide.md)
- [docs/cuda_native.md](docs/cuda_native.md)
- [docs/custom_components.md](docs/custom_components.md)
- [docs/model_artifacts.md](docs/model_artifacts.md)
- [docs/guide_project_structure.md](docs/guide_project_structure.md)
- [docs/guide_autograd.md](docs/guide_autograd.md)
- [docs/guide_feature_expansion.md](docs/guide_feature_expansion.md)

### 聚焦型技術筆記

這些文件內容仍有用，但不適合作為第一份閱讀材料：

- [docs/guide_layout_debug.md](docs/guide_layout_debug.md)
- [docs/cuda_batchnorm2d_evaluation.md](docs/cuda_batchnorm2d_evaluation.md)
- [docs/cuda_native_phase5_rfc.md](docs/cuda_native_phase5_rfc.md)

### 歷史背景或報告文件

這些是參考材料，不是主要 onboarding 文件：

- [docs/comparison_report.md](docs/comparison_report.md)
- [docs/comparison_completion_report.md](docs/comparison_completion_report.md)
- [docs/optimization_progress.md](docs/optimization_progress.md)
- [docs/benchmark_report_template.md](docs/benchmark_report_template.md)
- [docs/agent_completion_report.md](docs/agent_completion_report.md)

## 快速導覽

如果你不知道下一步該看哪裡：

- 想看真實支援邊界： [docs/backend_capabilities.md](docs/backend_capabilities.md)
- 想理解同一份 shared config 如何跨 backend： [docs/dual_backend_guide.md](docs/dual_backend_guide.md)
- 想看 native build/debug 細節： [docs/guide_project_structure.md](docs/guide_project_structure.md) 與 [docs/guide_layout_debug.md](docs/guide_layout_debug.md)
- 想看 extension points： [docs/custom_components.md](docs/custom_components.md)
- 想看 checkpoint 格式與模型復用： [docs/model_artifacts.md](docs/model_artifacts.md)
- 想看實驗性 native graph backend： [docs/cuda_native.md](docs/cuda_native.md)

## 簡單原則

- `torch/flex` 是 reference implementation，也是新功能的第一站。
- `train-autograd` 適合當 CPU 側 correctness oracle，也適合框架學習。
- `cuda_native` 是主要 native 成長方向，但目前仍屬實驗性。
- `cuda_legacy` 請視為歷史維護路徑，只在 validator 定義的邊界內使用與修補。
