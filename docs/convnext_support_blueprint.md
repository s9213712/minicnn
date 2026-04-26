# ConvNeXt Support Blueprint

Last updated: 2026-04-23

This is the compact execution blueprint for MiniCNN's current ConvNeXt work.

It is intentionally narrower than a full roadmap. Use it when you need to
answer:

- what ConvNeXt support already exists
- what still does not exist
- what a future ConvNeXt patch is allowed to do
- what a future ConvNeXt patch must not claim

See also:

- [convnext_capability_matrix.md](convnext_capability_matrix.md)
- [convnext_yaml_ir_decision.md](convnext_yaml_ir_decision.md)
- [backend_capabilities.md](backend_capabilities.md)
- [capability_expansion_slices.md](capability_expansion_slices.md)

## Current Supported Slice

Today MiniCNN supports exactly this ConvNeXt-related slice:

- one built-in `ConvNeXtBlock` on `torch/flex`
- one tracked torch-only template: `templates/cifar10/convnext_like.yaml`
- one beta-grade `cuda_native` block smoke path
- one beta-grade `cuda_native` explicit primitive smoke path
- test-backed flex builder and registry coverage
- capability-table and docs coverage that explicitly mark backend boundaries

That is enough to call ConvNeXt a minimal experimental frontend path.

That is not enough to call ConvNeXt a repo-wide supported architecture family.

## Non-Goals

The current ConvNeXt slice does **not** include:

- `autograd` execution support
- `cuda_legacy` execution support
- `DropPath` / stochastic depth
- stage macros or a dedicated ConvNeXt YAML DSL
- ConvNeXt-specific IR lowering
- a family of ConvNeXt variants

Any patch that implies one of those without the required validation and tests is
out of scope.

## Allowed Near-Term Work

If ConvNeXt work resumes later, safe follow-up changes are limited to:

1. small quality improvements to `ConvNeXtBlock` or the native composite implementation
2. one additional backend-honest example or template if it proves a clearly new use case
3. stronger tests around the existing declaration surface
4. doc clarifications that keep backend boundaries explicit

These are still Slice B style changes in
[capability_expansion_slices.md](capability_expansion_slices.md):
frontend breadth without fake native parity.

## Disallowed Shortcuts

Do not do these as the next ConvNeXt step:

1. claim that `cuda_native` "almost supports" ConvNeXt
2. add `model.convnext` or any ConvNeXt-only schema branch
3. add IR structure just to mirror one frontend block
4. add native-facing docs before native validators and tests exist
5. merge multiple architecture experiments into one oversized patch

These shortcuts would make MiniCNN's public surface less honest.

## Reentry Conditions

ConvNeXt-specific work should only be reopened beyond the current slice if at
least one of these becomes true:

1. `torch/flex` needs more than one ConvNeXt-like template and duplication becomes costly
2. a generic frontend feature needed by multiple architectures is discovered through the ConvNeXt work
3. `cuda_native` reaches the generic prerequisites for ConvNeXt-like execution
   such as normalization, explicit primitive support, and later branching-graph support
4. compiler/IR work gains real backend consumers beyond torch-only introspection

If none of those conditions is true, ConvNeXt should remain in maintenance mode.

## Future Reentry Order

If the project later reopens ConvNeXt, the order should be:

1. restate the target backend and scope explicitly
2. decide whether the change is still `torch/flex`-only
3. add or tighten validator behavior first when new backend claims are involved
4. add the smallest test-backed implementation slice
5. update capability tables and user-facing docs in the same patch

## Maintainer Rule

When reviewing ConvNeXt-related work, ask these questions in order:

1. Is this still only a `torch/flex` feature?
2. Does it require new schema, or can `model.layers[]` already express it?
3. Does it change any backend claim?
4. Are tests and capability docs updated in the same patch?

If any answer is unclear, the patch is not ready.
