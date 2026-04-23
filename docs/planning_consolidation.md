# Planning Consolidation

Last updated: 2026-04-23

This note records the current planning-document structure after consolidating
overlapping roadmap and master-command style notes.

## Canonical Planning Sources

These are now the tracked source-of-truth documents:

1. [master_roadmap.md](master_roadmap.md)
   Role: the canonical project roadmap and priority framing
2. [capability_expansion_slices.md](capability_expansion_slices.md)
   Role: staged implementation slices for future capability work
3. [next_priority_order.md](next_priority_order.md)
   Role: queue ordering and local planning triage history for the reconstructed
   `comments/next/*` backlog

## Supporting Planning Notes

These remain useful, but they are not the main source of truth:

- [optimization_progress.md](optimization_progress.md)
  Role: current technical direction and progress framing
- [generalization_roadmap.md](generalization_roadmap.md)
  Role: frontend breadth vs backend honesty
- [cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)
  Role: design guardrails for high-risk `cuda_native` extensions
- [cuda_batchnorm2d_evaluation.md](cuda_batchnorm2d_evaluation.md)
  Role: scoped maintenance note for one `cuda_legacy` extension topic

## Retired Planning Shapes

The following planning shapes are now considered retired as primary planning
units:

- broad "master command" documents that try to act as both roadmap and execution plan
- overlapping "advanced roadmap" variants that restate the same priorities with
  different wording
- local-only planning notes that are not represented in tracked docs

If an old local note resurfaces, it should be handled in one of two ways:

1. mine it for missing ideas and merge those into the canonical tracked docs
2. retire it explicitly rather than keeping parallel planning sources

## Practical Rule

When planning work now:

- update `master_roadmap.md` if the priority order changes
- update `capability_expansion_slices.md` if the implementation slicing changes
- update `next_priority_order.md` if backlog ordering changes

Do not create a new parallel roadmap unless the existing docs cannot represent
the new planning need.
