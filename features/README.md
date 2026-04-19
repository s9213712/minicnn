# Feature Isolation Area

This directory is for isolated feature research before code is promoted into the supported `src/minicnn/` package.

Rules:

- Create one folder per experiment, for example `features/native-cuda-backend/`.
- Keep prototypes, notes, throwaway scripts, and exploratory tests inside that feature folder.
- Do not make production CLI commands import from `features/` by default.
- If an experiment fails, delete or archive only that feature folder and keep `main` stable.
- If an experiment succeeds, move the supported code into `src/minicnn/`, move tests into `tests/`, and update README/docs.
- Use a Git branch for every feature. For large rewrites, prefer `git worktree`.

Recommended branch workflow:

```bash
git checkout -b feature/<name>
mkdir -p features/<name>
```

Recommended large-experiment workflow:

```bash
git worktree add ../minicnn-feature-<name> -b feature/<name>
```
