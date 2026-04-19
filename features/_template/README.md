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
- `python3 -m pytest -q` passes.
- Fallback path on `main` remains clear.
