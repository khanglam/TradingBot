---
name: sync-branches
description: Commit pending changes on main, sync harness files to both campaign worktrees, then push all three branches (main, autoresearch/crypto, autoresearch/stocks) to origin. Run this any time you change a harness file on main and want campaign branches to reflect it. Also use it after staging cleanup changes on the worktrees (e.g. git rm). The skill commits, syncs, and pushes — ask the user what commit message to use on main if not already provided.
---

# /sync-branches

Commit and synchronize all three branches so they are consistent. Run from the
project root on the `main` branch.

## What this skill does (in order)

1. Validate prerequisites
2. Commit staged/modified changes on `main`
3. Sync the 6 harness files from `main` → both campaign worktrees
4. Commit any changes (staged or from the sync) on each campaign worktree
5. Push `main`, `autoresearch/crypto`, `autoresearch/stocks` to `origin`
6. Print a final summary

---

## Step 1 — Validate prerequisites

Run these checks. Abort (with a clear message) if any fail.

- **On main**: `git rev-parse --abbrev-ref HEAD` must return `main`. If not: "Run /sync-branches from the main branch checkout."
- **Worktrees exist**: Both `.worktrees/crypto` and `.worktrees/stocks` must exist as directories. If missing: "Run /init-local-dev first to create the campaign worktrees."
- **No merge conflicts**: `git status --porcelain` must not contain lines starting with `UU`, `AA`, or `DD`. If it does: "Resolve merge conflicts before syncing."
- **Campaign worktrees are on the right branch**: `git -C .worktrees/crypto rev-parse --abbrev-ref HEAD` must be `autoresearch/crypto`, and same for stocks. If not: abort with the branch names so the user can fix it.

---

## Step 2 — Commit changes on `main`

Check `git status --porcelain` on the root repo.

**If there are staged or unstaged changes:**
- If the user provided a commit message (e.g., passed as an argument to the skill), use it.
- If not, ask the user for a commit message before proceeding. Do not invent one silently.
- Stage all modified/new tracked files relevant to the project: `git add` the specific files that changed. Never use `git add .` or `git add -A` — that risks catching `.env`, build artifacts, or unintended files. Stage only files that belong to the project (harness files, workflow YAMLs, configs.toml, AGENTS.md, strategy files, etc.).
- Commit with the provided message.

**If the working tree is already clean:** skip with "[skip] main: nothing to commit".

---

## Step 3 — Sync harness files from `main` to both campaign worktrees

These are the ONLY files that get synced. Do not add others.

```
loop.py  backtest.py  program.md  data_fetch.py  requirements.txt  AGENTS.md
```

For each campaign in `[crypto, stocks]` (can run in parallel):

```bash
git -C .worktrees/<campaign> checkout main -- \
  loop.py backtest.py program.md data_fetch.py requirements.txt AGENTS.md
```

This overwrites those files in the worktree with the versions from `main`.
If a file hasn't changed, `git checkout main --` is a no-op for that file.

---

## Step 4 — Commit on each campaign worktree

For each campaign worktree (`crypto`, `stocks`):

Check `git -C .worktrees/<campaign> status --porcelain`.

**If there are staged changes** (from Step 3 sync OR from pre-existing staged changes like `git rm` deletions):

Get the current short SHA of main: `MAIN_SHA=$(git rev-parse --short=7 HEAD)`

Commit with the message:
```
chore: sync harness from main <MAIN_SHA>
```

If the worktree has staged changes that are NOT from the harness sync (e.g., pre-staged `git rm` deletions from a cleanup), include them in the same commit — they are already staged and belong in this commit. Do not unstage them.

**If the working tree is already clean:** skip with "[skip] <campaign>: nothing to commit".

---

## Step 5 — Push all three branches

Push in this order:
1. `git push origin main`
2. `git -C .worktrees/crypto push origin HEAD:autoresearch/crypto`
3. `git -C .worktrees/stocks push origin HEAD:autoresearch/stocks`

If any push fails with a non-fast-forward error, stop and tell the user:
"Push rejected for <branch> — someone pushed ahead. Run `git pull --rebase origin <branch>` in that worktree, then re-run /sync-branches."

Do NOT force push under any circumstance.

---

## Step 6 — Print a final summary

```
Sync complete:
  ✓ main              committed "<message>" @ <sha>   (or "[skip] nothing to commit")
  ✓ autoresearch/crypto   committed @ <sha>           (or "[skip] nothing to commit")
  ✓ autoresearch/stocks   committed @ <sha>           (or "[skip] nothing to commit")
  ✓ pushed             main, autoresearch/crypto, autoresearch/stocks → origin

Harness files synced:
  loop.py  backtest.py  program.md  data_fetch.py  requirements.txt  AGENTS.md
```

---

## Behavior rules

- **Never use `git add .` or `git add -A`.** Always stage specific files by name.
- **Never force push.** If a push is rejected, stop and explain.
- **Never touch `strategies/`** during the harness sync. The loop owns strategy files on campaign branches; main owns the frozen copies. The sync NEVER copies strategies between branches.
- **Never sync `configs.toml` to campaign branches.** It lives only on `main` and is read via `git show origin/main:configs.toml` at runtime.
- **Never sync `app.py`, `scan.py`, `live_trade.py`, `sync_branches.py`, `web/`, `.github/`, or `.claude/`** to campaign branches.
- **Ask before committing on `main`** if no commit message was provided. Do not silently invent messages for main. Campaign sync commits always use the `chore: sync harness from main <sha>` format.
- **AGENTS.md rule**: if any file was added, removed, or moved as part of the changes being committed, verify that `AGENTS.md` has been updated to reflect it. If not, update `AGENTS.md` before committing.
