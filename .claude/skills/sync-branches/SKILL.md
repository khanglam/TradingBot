---
name: sync-branches
description: Merge dev into main and push (same as sync_branches.yml). Run from a main checkout after research on dev.
---

# /sync-branches

Merge `dev` into `main` locally — mirrors `.github/workflows/sync_branches.yml`.

## Steps

1. `git rev-parse --abbrev-ref HEAD` must be `main`.
2. `git fetch origin dev`
3. `git merge origin/dev --no-edit -m "merge dev into main"`
4. Resolve conflicts if any, then `git push origin main`

Do not force push.
