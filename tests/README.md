# MiniCNN Test Layout

The `tests/` folder is organized by responsibility, not by execution order.

## Core Families

- `test_phase*.py`: feature-by-feature regression coverage for the staged expansion work
- `test_cuda_native_phase*.py`: `cuda_native` backend milestones and integration coverage
- `test_flex_*.py`, `test_unified_*.py`, `test_training_loop.py`: active frontend and training-path behavior
- `test_runtime_compiler_fusion.py`: compiler and runtime inference path

## Regression Buckets

These files used to have historical review-oriented names such as
`test_comment_fixes.py` and `test_review_fixes.py`. They were renamed so their
purpose is visible from the filename:

- `test_regressions_cli_runtime.py`: CLI, build, checkpoint, and runtime regressions
- `test_regressions_training_contracts.py`: training-contract and validator regressions
- `test_regressions_tensor_ops.py`: tensor-op, optimizer, and math regressions

## What Is Not Dead Weight

- `test_comment_fixes.py`, `test_new_findings_fixes.py`, and `test_review_fixes.py`
  are no longer present as separate historical buckets.
- The replacement regression files still carry real coverage; they were renamed,
  not removed.
- `test_phase12_docs.py` is intentional. It protects the config/doc examples and
  should stay as long as the docs are part of the public surface.
