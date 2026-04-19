#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/s9213712/minicnn.git"
DEFAULT_BRANCH="main"

if [ ! -d .git ]; then
  git init
fi

git add .
git commit -m "feat: initial MiniCNN release" || true
git branch -M "$DEFAULT_BRANCH"

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REPO_URL"
fi

echo "Then run: git push -u origin $DEFAULT_BRANCH"
