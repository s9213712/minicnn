# Phase 1–3 Execution Plan

Last updated: 2026-04-23

This is the tracked execution brief for the merged V1–V3 backlog.

Goals:

1. templates should not ship in a broken-by-default state
2. `minicnn smoke` should distinguish optional torch/runtime availability from
   real repo breakage
3. Windows native build docs should match actual output locations and helper UX
4. CLI failures should point users toward a fix instead of dropping raw context
5. a new user should be able to reach the first successful training run quickly

## Phase 1: Template And Dataset Validity

Scope:

- fix invalid MNIST template splits
- add a template-validity regression test over `templates/**/*.yaml`
- document dataset split rules in `templates/README.md` and `USAGE.md`

Acceptance:

- `minicnn train-flex --config templates/mnist/lenet_like.yaml` starts without
  a split-related `ValueError`
- template-validity tests pass

## Phase 2: Smoke And Dependency Reporting

Scope:

- keep `torch` optional for smoke/doctor/healthcheck
- expose machine-readable torch/cuda/native status in smoke output
- improve smoke severity rules so missing optional pieces become warnings, not
  misleading repo-failure signals
- improve user-facing error messages where the cause and fix are known

Acceptance:

- `minicnn smoke --format json` reports torch/cuda/native availability
- a no-torch environment does not crash `minicnn smoke`

## Phase 3: Windows Build Parity

Scope:

- align `docs/guide_windows_build.md` with actual build output paths
- upgrade `scripts/build_windows_native.ps1` success output and next-step hints
- optionally suggest a recommended CUDA architecture from detected GPU info

Acceptance:

- the documented DLL output locations match the helper's real build directories
- the Windows build helper prints the built DLL paths and clear next steps

## Non-Negotiable Rules

- broken templates must not remain as the default tracked examples
- optional dependencies must not be reported as fatal repo corruption
- user-facing failures must include a plausible next action where possible
- docs must describe real paths and commands, not aspirational ones
