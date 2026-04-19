# GitHub Publish Guide

## 1. Create a new GitHub repository

Suggested name:

```text
minicnn
```

## 2. Initialize and push

```bash
git init
git add .
git commit -m "feat: initial MiniCNN release"
git branch -M main
git remote add origin https://github.com/s9213712/minicnn.git
git push -u origin main
```

## 3. Recommended repo settings

- Add a short description.
- Enable Issues.
- Enable Actions.
- Add topics such as `cuda`, `deep-learning`, `cnn`, `python`, `pytorch`.

## 4. First release

After your first stable push:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## 5. Optional improvements

- add screenshots
- add benchmark tables
- add example experiment outputs
- add package publishing later
