# Contributing to MiniCNN

Thanks for contributing.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[torch,dev]
```

## Before opening a pull request

Run:

```bash
python -m compileall -q src
pytest -q
```

Also do a short repo hygiene pass before pushing:

- scan changed docs/README files for stale backend wording
- check whether any production file became an oversized concentration point
- split concentrated runtime/helper code before it becomes review-hostile
- avoid committing runtime artifacts, caches, or generated binaries
- run `git diff --check`

## Style guidelines

- Prefer small, focused commits.
- Keep public CLI behavior documented in `README.md`.
- Add or update tests when changing framework behavior.
- Preserve the separation between user-facing config paths and framework internals.

## Project principles

- Users should be able to change model structure through configuration.
- New components should be easy to register and discover.
- Native CUDA migration should remain incremental and not break the flexible path.
