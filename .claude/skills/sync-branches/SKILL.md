---
name: sync-branches
description: Run the promotion gauntlet on main (fetch dev, sync_branches.py), commit and push main if a strategy was promoted. Run from main after daily loops on dev. Ask the user for a commit message if there are unrelated pending changes on main.
---

# /sync-branches

Promote winning strategies from `dev` to `main`. Run from a **`main`** checkout.

## What this skill does (in order)

1. Validate prerequisites
2. Commit any pending changes on `main` (if user provided a message)
3. `git fetch origin dev:dev`
4. Run `python sync_branches.py --campaign stocks` and `--campaign crypto`
5. If promotion changed files, commit and push `main`
6. Print summary

---

## Step 1 — Validate prerequisites

- **On main**: `git rev-parse --abbrev-ref HEAD` must be `main`.
- **dev exists on origin**: `git fetch origin dev` must succeed.
- **No merge conflicts** on main.

---

## Step 2 — Commit pending changes on main (optional)

If `git status --porcelain` is non-empty on main, ask for a commit message (unless user provided one). Stage specific files only — never `git add .`.

If clean: `[skip] main: nothing to commit before promote`.

---

## Step 3 — Fetch dev and run promotion

```bash
git fetch origin dev:dev
python sync_branches.py --campaign stocks
python sync_branches.py --campaign crypto
```

Exit code `0` = promoted (caller should commit). Exit code `1` = no-op. Exit code `2` = error — stop.

---

## Step 4 — Commit + push if promoted

For each campaign that returned `0`:

```bash
SHA=$(git rev-parse --short=7 dev)
git add strategies/<campaign>.py
git commit -m "promote <campaign> ${SHA}" --no-verify
```

Then `git push origin main` (rebase retry up to 3 times if needed).

---

## Behavior rules

- **Never force push.**
- **Never copy harness files** — only `strategies/<campaign>.py` promotes.
- **Harness parity:** if `backtest.py` on main lags dev, merge `dev` → `main` separately before trusting promotion results.
- Update `AGENTS.md` if the commit changes repo structure.
