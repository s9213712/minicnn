# Feature: <name>

## Goal

Describe the feature and the problem it solves.

## Isolation Boundary

List which files are experimental and which production modules may eventually receive promoted code.

## Prototype Commands

```bash
# Add local prototype commands here.
```

## Promotion Checklist

- Prototype behavior is understood.
- Supported implementation moved into `src/minicnn/`.
- Tests moved or added under `tests/`.
- README/docs updated.
- Template docs/comments updated when user-facing training commands or outputs change.
- Config parsing still uses strict booleans, `train.init_seed`, and list-index CLI overrides rather than feature-local parsing shortcuts.
- `python3 -m pytest -q` passes.
- `python3 -m compileall -q src/minicnn tests` passes.
- Native changes rebuild with `minicnn build --legacy-make --variant both --check`.
- Fallback path on `main` remains clear.
