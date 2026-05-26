---
name: init-local-dev
description: Bootstrap (or repair) a local TradingBot dev environment idempotently. Creates the Python virtualenv, installs requirements, fetches the dev branch, and checks out dev for research. OS-agnostic — works on macOS, Linux, and Windows.
---

# /init-local-dev

Set up everything a developer needs to run TradingBot locally: virtualenv, dependencies, and the `dev` branch checkout. Every step is idempotent.

## Detect environment first

Before doing anything, capture:

- **Project root** = the current working directory (must contain `loop.py`, `app.py`, and a `.git` directory). If not, abort with a clear message.
- **OS** = inspect via `uname -s` on macOS/Linux or check `$OS` / Python's `platform.system()` on Windows.
- **Python interpreter** = whichever of `python3`, `python`, or `py -3` resolves first. Verify version ≥ 3.11.

## The steps (run sequentially, each idempotent)

### 1. Confirm working tree is clean

Run `git status --porcelain`. If non-empty, stop: "Working tree has uncommitted changes; commit or stash before running setup."

### 2. Fetch and checkout `dev`

```bash
git fetch origin dev
```

If `origin/dev` does not exist, stop with: "Create the dev branch on origin first — see MIGRATION.md."

If local HEAD is not `dev`:
```bash
git checkout dev 2>/dev/null || git checkout -b dev origin/dev
```

### 3. Create the virtualenv (skip if `.venv/` exists)

Run `<python> -m venv .venv` if missing.

### 4. Install requirements (skip if importable)

Use `.venv/bin/pip` or `.venv\Scripts\pip.exe`. Pre-check: `import fastapi, backtesting, ccxt, openai`.

### 5. Ensure `data/` exists

`mkdir -p data` (or `mkdir data` on Windows) if missing.

### 6. Verify `.env` (don't create)

Tell user to copy `.env.example` if `.env` is missing.

### 7. Print final summary

```
Setup complete:
  ✓ branch            dev @ <sha>
  ✓ .venv             Python 3.x at <path>
  ✓ requirements
  ✓ data/

Next:
  - Fetch OHLCV: <venv-python> data_fetch.py ...
  - Dashboard:   <venv-python> app.py   (from dev)
  - Loop:        <venv-python> loop.py --iters 10
```

## Behavior rules

- **Idempotent.** Re-running should mostly show `[skip]` lines.
- **No destructive git ops** (no reset, no force push).
- **No git pushes.**
- **Optional:** if `.worktrees/` exists from the old layout, tell the user to remove it per MIGRATION.md — do not auto-delete.
