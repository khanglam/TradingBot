---
name: init-local-dev
description: Bootstrap (or repair) a local TradingBot dev environment idempotently. Creates the Python virtualenv with uv, installs requirements, fetches the dev branch, and checks out dev for research. OS-agnostic — works on macOS, Linux, and Windows.
---

# /init-local-dev

Set up everything a developer needs to run TradingBot locally: uv, virtualenv, dependencies, and the `dev` branch checkout. Every step is idempotent.

## Detect environment first

Before doing anything, capture:

- **Project root** = the current working directory (must contain `loop.py`, `app.py`, and a `.git` directory). If not, abort with a clear message.
- **OS** = inspect via `uname -s` on macOS/Linux or check `$OS` / Python's `platform.system()` on Windows.
- **`uv`** = must be on PATH (`uv --version`). If missing, tell the user to install from https://docs.astral.sh/uv/getting-started/installation/

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

### 3. Ensure Python 3.11+ (via uv)

Run `uv python install 3.12` if no suitable interpreter is available. Optionally `uv python pin 3.12` in the project root.

### 4. Create the virtualenv (skip if `.venv/` exists)

Run `uv venv .venv --python 3.12` if missing.

### 5. Install requirements (skip if importable)

Run `uv pip install -r requirements.txt`. Pre-check: `uv run python -c "import fastapi, backtesting, ccxt, openai"`.

### 6. Ensure `data/` exists

`mkdir -p data` (or `mkdir data` on Windows) if missing.

### 7. Verify `.env` (don't create)

Tell user to copy `.env.example` if `.env` is missing.

### 8. Print final summary

```
Setup complete:
  ✓ branch            dev @ <sha>
  ✓ .venv             Python 3.x at <path>
  ✓ requirements
  ✓ data/

Next:
  - Fetch OHLCV: uv run python data_fetch.py ...
  - Dashboard:   uv run python app.py   (from dev)
  - Loop:        uv run python loop.py --iters 10
```

## Behavior rules

- **Idempotent.** Re-running should mostly show `[skip]` lines.
- **No destructive git ops** (no reset, no force push).
- **No git pushes.**
- **Optional:** if `.worktrees/` exists from the old layout, tell the user to remove it per MIGRATION.md — do not auto-delete.
