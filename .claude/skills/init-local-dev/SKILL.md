---
name: init-local-dev
description: Bootstrap (or repair) a local TradingBot dev environment idempotently. Creates the Python virtualenv, installs requirements, fetches the autoresearch branches, sets up the campaign git worktrees, and links the shared data/ cache. Each step is checked first and skipped if already done. OS-agnostic — works on macOS, Linux, and Windows.
---

# /init-local-dev

Set up everything a developer needs to run TradingBot locally: virtualenv, dependencies, autoresearch branches, campaign worktrees, and the shared `data/` symlinks. Every step is idempotent — re-running the skill on a partially-set-up machine fixes only what's missing.

## Detect environment first

Before doing anything, capture:

- **Project root** = the current working directory (must contain `loop.py`, `app.py`, and a `.git` directory). If not, abort with a clear message.
- **OS** = inspect via `uname -s` on macOS/Linux or check `$OS` / Python's `platform.system()` on Windows. Use this to pick `python` vs `python.exe` paths and symlink vs junction commands.
- **Python interpreter** = whichever of `python3`, `python`, or `py -3` resolves first. Verify version ≥ 3.11 (the project assumes modern Python). If too old, abort with a clear message asking the user to install Python 3.11+.

Do not invent OS or Python detection — actually run the commands and read the output.

## The steps (run sequentially, each idempotent)

### 1. Confirm working tree is clean

Run `git status --porcelain`. If the output is non-empty, **stop** and tell the user: "Working tree has uncommitted changes; commit or stash before running setup." Do not modify a dirty tree.

### 2. Ensure we're on `main`

Run `git rev-parse --abbrev-ref HEAD`. If it isn't `main`, tell the user "Run from the main checkout (`git checkout main`), then retry." Stop.

### 3. Create the virtualenv (skip if `.venv/` already exists)

Check if `.venv/` exists at the project root.
- If yes: skip with a "[skip] .venv already exists" message.
- If no: run `<python> -m venv .venv` where `<python>` is the interpreter from detection. Confirm the resulting `.venv/bin/python` (Mac/Linux) or `.venv\Scripts\python.exe` (Windows) is executable.

### 4. Install requirements (skip if all packages already importable)

Use the venv's pip:
- macOS/Linux: `.venv/bin/pip`
- Windows: `.venv\Scripts\pip.exe`

Quick pre-check: try `<venv-python> -c "import fastapi, backtesting, ccxt, openai"`. If it succeeds, treat requirements as installed and skip with "[skip] requirements already installed".

Otherwise run `<venv-pip> install -r requirements.txt`. Surface only the final summary line, not the full install log.

### 5. Fetch the autoresearch branches from origin

Run `git fetch origin autoresearch/stocks autoresearch/crypto`. If origin doesn't have one or both branches, surface the error and stop with: "Push autoresearch branches first: `git push -u origin autoresearch/stocks autoresearch/crypto` from a machine that has them."

If the local refs `autoresearch/stocks` and `autoresearch/crypto` don't exist yet, create them tracking origin:

```
git branch --track autoresearch/stocks origin/autoresearch/stocks   # if missing
git branch --track autoresearch/crypto origin/autoresearch/crypto   # if missing
```

Use `git show-ref --verify --quiet refs/heads/<name>` to test for existence before creating.

### 6. Create the campaign worktrees (skip if already present)

For each campaign in `[stocks, crypto]`:

- Check `git worktree list --porcelain` for an entry matching `.worktrees/<campaign>`.
- If present: skip with "[skip] worktree .worktrees/<campaign> already exists".
- If absent and the directory `.worktrees/<campaign>` already exists on disk (orphaned, e.g. someone `rm`'d the worktree without `git worktree remove`): run `git worktree prune` first, then proceed.
- Create with: `git worktree add .worktrees/<campaign> autoresearch/<campaign>`.

### 7. Create the shared `data/` link inside each worktree

Background: each worktree starts without a `data/` directory because `data/` is gitignored. We link each worktree's `data/` to the main checkout's `data/` so the OHLCV cache is shared (one fetch, three views).

For each campaign in `[stocks, crypto]`:

- If `<root>/data/` doesn't exist yet, create it: `mkdir -p data` (macOS/Linux) or `mkdir data` (Windows). Empty is fine — the user runs `data_fetch.py` later.
- Check if `.worktrees/<campaign>/data` already exists.
  - If it's already a symlink/junction pointing at `<root>/data`: skip with "[skip] data link already in place for <campaign>".
  - If it exists but isn't a link: warn the user "`.worktrees/<campaign>/data` exists but is not a link; remove it manually and re-run." Do NOT auto-delete.
  - If absent: create the link.
    - **macOS / Linux**: `ln -s "<absolute-path-to-root>/data" .worktrees/<campaign>/data`
    - **Windows** (cmd or PowerShell): `mklink /J ".worktrees\<campaign>\data" "<absolute-path-to-root>\data"`
      - Use a directory junction (`/J`), not a symbolic link, so admin privileges aren't required.

Always pass an absolute path to the link target. Relative paths break when `cwd` differs (e.g., when the loop subprocess runs from inside the worktree).

### 8. Verify `.env` (don't create — just check)

If `.env` doesn't exist at the project root, tell the user: "No `.env` file found. Copy from `.env.example` and fill in `OPENROUTER_API_KEY` (and `ALPACA_*` if you want local paper trading)." Do not create it for them — secrets are theirs to manage.

If `.env` exists but is missing `OPENROUTER_API_KEY`, mention it but don't fail.

### 9. Print a final summary

After all steps, print a concise summary:

```
Setup complete:
  ✓ .venv             (Python 3.x.y at <path>)
  ✓ requirements      (N packages)
  ✓ branches          autoresearch/stocks @ <sha>, autoresearch/crypto @ <sha>
  ✓ worktrees         .worktrees/stocks, .worktrees/crypto
  ✓ data link         .worktrees/stocks/data → <root>/data, same for crypto
  ! .env              missing — see .env.example   (only if applicable)

Next:
  - Fetch OHLCV data: <venv-python> data_fetch.py --asset stocks --symbol TSLA --timeframe 1d --start 2015-01-01
  - Launch dashboard: <venv-python> app.py
```

Use `✓` for completed steps, `!` for warnings, `✗` for errors. Replace placeholders with real values from the environment.

## Behavior rules

- **Idempotent throughout.** Every step checks current state before acting. Re-running on a fully-set-up machine should produce only "[skip]" lines and the final summary.
- **No destructive operations.** Don't delete pre-existing files. Don't run `git reset`. Don't rewrite `.env`. If state is ambiguous (e.g., `data/` exists as a real directory inside a worktree), warn and stop rather than guess.
- **Surface progress.** Print a one-line `[N/9] <step name>` heading before each step. Keep tool output short — long install logs should be summarized to the final result line.
- **Stop on hard errors.** A missing branch on origin, a dirty tree, or wrong Python version aborts the whole skill cleanly. Don't try to push past prerequisites.
- **No git pushes.** This skill never pushes. It only sets up the local environment.
- **No CI changes.** Don't edit workflows or `MIGRATION.md`. The skill is a local-only convenience; nothing it does affects shared state.
