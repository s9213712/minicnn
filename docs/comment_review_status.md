# Comment Review Status

This file records the result of processing `comments/minicnn_full_review_and_refactor_spec.txt`.

## Completed

| Area | Status |
|---|---|
| Lazy CUDA loading | Already implemented through `get_lib()` and `LazyCudaLibrary`; added `check_cuda_ready()` diagnostics. |
| Settings/config overrides | Already implemented through schema/YAML/CLI overrides; added shared scalar parsing and `settings.summarize()`. |
| CLI introspection | Added `doctor`; expanded `info`; added `train-cuda`, `train-torch`, and `compare`. |
| Package hygiene | `.gitignore` already covers caches, artifacts, native libs, build outputs, and model checkpoints. |
| CUDA training split | Existing `cuda_workspace.py` and `cuda_ops.py` are retained; added `cuda_epoch.py` for augmentation/epoch-level helpers. |
| Explicit device state | Added `DeviceWeights` and `VelocityBuffers` dataclasses while keeping tuple iteration compatibility for legacy call sites. |
| Duplicated eval forward path | Centralized CUDA evaluation forward flow through `_forward_logits_ptr()`. |
| Checkpoint reload safety | `reload_weights_from_checkpoint()` now uploads replacement weights before freeing the existing runtime weights. |
| Diagnostics tests | Added tests for CLI command exposure, transactional reload, workspace int buffers, empty evaluation, parser sharing, and CUDA crop behavior. |
| Docs | README, Traditional Chinese README, USAGE, and this status report document the new commands and remaining boundary. |

## Deferred

| Item | Reason |
|---|---|
| Full class-based CUDA trainer rewrite | The current legacy trainer is now safer and more modular, but a complete class-based rewrite would be a larger behavior-preserving migration requiring GPU regression runs. |
| Full `train_one_epoch()` extraction | The inner CUDA forward/backward/update loop is still in `train_cuda.py` to avoid introducing untested state transfer bugs; `cuda_epoch.py` now hosts lower-risk epoch helpers. |
| Full native autograd stack with Conv2d/MaxPool2d/BatchNorm2d | MiniCNN now has a minimal CPU/NumPy autograd core, but full CNN autograd and CUDA-autograd bridging remain long-term work. |
| CUDA-autograd bridge | Requires a separate design pass to map graph `Function` objects to existing CUDA kernels without overclaiming backend support. |
| `src/minicnn/features/` package experiments | The repository already uses top-level `features/` for isolated prototypes; moving experiments into the package would make import boundaries less clear. |
| Single `minicnn train` command | The current supported commands are `train-dual`, `train-flex`, `train-torch`, and `train-cuda`; adding another alias is optional and not needed for current docs. |

## Verification

Run after this review:

```bash
PYTHONPATH=src python -m pytest -q tests
python -m compileall -q src
git diff --check
```
