# Branch migration (completed)

The repo now uses **`main`** (frozen strategies + paper) and **`dev`** (research loop).

## Daily workflow

```bash
git checkout dev    # research, dashboard, loop.py
git checkout main   # frozen strategies, paper trading locally
```

Avoid running the loop locally at **03:00 PST** when GitHub Actions pushes to `dev`.

## What was migrated

- `dev` seeded from `autoresearch/stocks` and `autoresearch/crypto` (strategies + `results/*.tsv`)
- Local worktrees under `.worktrees/` removed
- Remote branches `autoresearch/stocks` and `autoresearch/crypto` deleted

If you still have stale local remote-tracking refs, run: `git fetch --prune origin`
