# Next Priority Order

Branch: `next-priority-planning`
Date: 2026-04-23

This file reconstructs and prioritizes the old local `comments/next/*` queue.
That folder was previously local-only and ignored by git, so the ranking below
is based on the last observed filename set before cleanup plus the current state
of `main`.

## Rules Used For Ranking

- `Importance`: how much the item improves the real product, developer clarity,
  or user onboarding if completed well
- `Difficulty`: estimated implementation and integration cost
- `Priority`: execution order, favoring high importance with manageable risk

Scales:

- `Importance`: 1 low -> 5 high
- `Difficulty`: 1 low -> 5 high

## Already Landed In `main`

These old `next` items should not stay in the active queue:

1. `0.backend_role_docs_patch_command.md`
   Status: already reflected in README, backend-role docs, and completion docs.
2. `0.minicnn_backend_role_repositioning_command.md`
   Status: already landed.
3. `minicnn_show_model_show_graph_cli_skeleton_command.md`
   Status: `show-model` and `show-graph` are now real CLI features, not
   skeleton placeholders.

## Active Priority Order

### P1

1. `4._minicnn_future_master_roadmap_merged.txt`
   Importance: 5
   Difficulty: 2
   Status: completed on `next-priority-planning` via `docs/master_roadmap.md`
   Why first: this is the best candidate for the next canonical planning source.
   It should be distilled into one tracked roadmap and used to delete overlap
   between the remaining master-command and roadmap notes.

2. `MiniCNN CNN Example Expansion Roadmap/3_minicnn_integration_order_checklist.md`
   Importance: 5
   Difficulty: 3
   Status: completed on `next-priority-planning` via `examples/README.md`, `templates/README.md`, and `docs/example_expansion_plan.md`
   Why second: examples are the shortest path from "repo exists" to "user can
   learn it quickly." The current repo already has working CLI and docs; the
   next highest-ROI improvement is better example coverage and a clean example
   integration order.

3. `1.minicnn_full_agent_instruction_capability_expansion.md`
   Importance: 5
   Difficulty: 4
   Status: completed on `next-priority-planning` via `docs/capability_expansion_slices.md`
   Why third: this likely drives the next real feature-growth wave, but it is
   broader and riskier than example expansion. It should be trimmed into a
   staged capability plan instead of being executed as one giant instruction.

### P2

4. `3.minicnn_advanced_roadmap.txt`
   Importance: 4
   Difficulty: 3
   Status: completed on `next-priority-planning` via `docs/planning_consolidation.md`
   Why here: useful, but it overlaps with the merged-future-roadmap item above.
   It should be mined for missing ideas, then either merged into the canonical
   roadmap or retired.

5. `2.minicnn_v6_master_command.txt`
   Importance: 4
   Difficulty: 3
   Status: completed on `next-priority-planning` via `docs/planning_consolidation.md`
   Why here: probably an execution wrapper, not the source of truth. It should
   be rewritten only after the roadmap is stabilized, otherwise it will encode
   outdated assumptions.

6. `MiniCNN CNN Example Expansion Roadmap/3_minicnn_cnn_examples_phased_command_book_flex_cuda_native.md`
   Importance: 4
   Difficulty: 3
   Status: completed on `next-priority-planning` via `docs/example_expansion_plan.md`
   Why here: high onboarding value, but it depends on first deciding the example
   integration order and which backends/examples should be treated as canonical.

### P3

7. `ConvNeXt/3_convnext_backend_capability_matrix.md`
   Importance: 3
   Difficulty: 4
   Status: completed on `convnext-capability-planning` via `docs/convnext_capability_matrix.md`
   Why here: capability-matrix work is useful only after the broader capability
   plan is made explicit. Doing ConvNeXt-specific matrix work too early risks
   optimizing for one architecture before the general frontend/backend boundary
   questions are settled.

8. `ConvNeXt/3_convnext_config_and_tests.md`
   Importance: 3
   Difficulty: 5
   Status: completed on `convnext-capability-planning` via `ConvNeXtBlock`,
   `templates/cifar10/convnext_like.yaml`, and flex/docs test coverage
   Why here: this is real implementation work with meaningful regression risk.
   It should wait until the capability-expansion brief has been reduced to a
   narrower staged scope.

9. `ConvNeXt/3_minicnn_convnext_yaml_spec_and_ir_schema_draft.md`
   Importance: 3
   Difficulty: 5
   Status: completed on `convnext-capability-planning` via
   `docs/convnext_yaml_ir_decision.md` with an explicit "no new schema yet"
   decision
   Why here: YAML and IR schema expansion is architecture-shaping work. It is
   not the next best move while the project is still deciding how much frontend
   breadth should be promoted ahead of native support.

10. `ConvNeXt/3_minicnn_convnext_support_blueprint_and_agent_command.md`
    Importance: 2
    Difficulty: 4
    Status: completed on `convnext-capability-planning` via
    `docs/convnext_support_blueprint.md`
    Why here: this is planning support for a feature that is already lower
    priority than examples and general capability expansion.

## Asset Packs And Supporting Files

These should not lead the queue by themselves. They are attachments, not the
primary planning units:

- `ConvNeXt/3_minicnn_convnext_skeleton_pack.zip`
- `MiniCNN CNN Example Expansion Roadmap/3_all_examples_support_pack.zip`
- `MiniCNN CNN Example Expansion Roadmap/configs_examples_pack.zip`

Use them only after the corresponding roadmap/checklist item becomes active.

## Recommended Execution Sequence

1. Turn `4._minicnn_future_master_roadmap_merged.txt` into one tracked roadmap
   doc under `docs/`.
2. Pull the example-expansion checklist into a tracked execution plan and decide
   which examples are the canonical beginner path.
3. Reduce `1.minicnn_full_agent_instruction_capability_expansion.md` into
   staged implementation slices with explicit validator/test/doc boundaries.
4. Merge or retire the overlapping roadmap/master-command notes.
5. Revisit ConvNeXt only after the general capability-expansion scope is
   stable.

## Short Version

If only one thing starts now, do this:

1. canonicalize the merged future roadmap
2. improve the example-expansion path
3. only then start broader capability-expansion work
