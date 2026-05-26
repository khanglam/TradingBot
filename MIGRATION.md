# Migration: autoresearch branches → main + dev

One-time steps after pulling this refactor.

## 1. Create the `dev` branch on origin

From your machine, with a clean tree on `main`:

```bash
# Option A: branch dev from main, then copy latest research strategies
git checkout main
git pull origin main
git checkout -b dev

# Copy candidates from old campaign branches if they exist locally:
git fetch origin autoresearch/stocks autoresearch/crypto 2>/dev/null || true
git show origin/autoresearch/stocks:strategies/stocks.py > strategies/stocks.py 2>/dev/null || true
git show origin/autoresearch/crypto:strategies/crypto.py > strategies/crypto.py 2>/dev/null || true

git add strategies/ results/
git commit -m "chore: seed dev from campaign branches"
git push -u origin dev
```

If you have no `autoresearch/*` remotes, `git checkout -b dev && git push -u origin dev` from current `main` is enough.

## 2. Remove local worktrees

```bash
git worktree remove .worktrees/stocks --force 2>/dev/null || true
git worktree remove .worktrees/crypto --force 2>/dev/null || true
git worktree prune
# Remove empty directory (Windows: rmdir /s /q .worktrees)
```

## 3. Daily workflow

```bash
git checkout dev    # research, dashboard, loop.py
# ... edit, run loop locally (avoid 03:00 PST if GHA is scheduled) ...

git checkout main   # review frozen strategies, run paper locally
```

## 4. Delete obsolete remote branches

After verifying GHA loops and promotion work on `dev` / `main`:

```bash
git push origin --delete autoresearch/stocks
git push origin --delete autoresearch/crypto
```

Update branch protection rules in GitHub Settings if they referenced the old branches.

## 5. GitHub Actions

- Disable or ignore old workflow runs for `loop-campaign.yml` and `scan.yml` (removed).
- Ensure `dev` exists on origin before the first scheduled `loop-stocks.yml` / `loop-crypto.yml` run.
