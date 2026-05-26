# Agent Guidelines — TradingBot

This file is the authoritative reference for repository structure and branch ownership.
**Rule: any time a file is added, moved, renamed, or deleted on any branch, update this file
to reflect the change before committing.**

---

## Repository Purpose

Karpathy-style autoresearch optimizer for trading strategies. The loop mutates strategies
on `dev`. Once a day, `sync_branches.yml` merges `dev` → `main` for paper trading.

---

## Two-Branch Structure

### `main` — deployment surface

Never mutated by the loop. Updated by the daily **merge dev → main** workflow and
manual commits. `paper.yml` and `live_trade.py` read from here.

| File / Dir | Purpose |
|---|---|
| `live_trade.py` | Alpaca paper/live executor |
| `configs.toml` | Campaign config — **lives on main**; dev CI reads `origin/main:configs.toml` |
| `strategies/*.py` | What paper trades — copy of dev after daily merge |
| `.github/workflows/` | CI definitions (read from default branch) |

### `dev` — research branch

All loop work happens here.

| File / Dir | Purpose |
|---|---|
| `strategies/*.py` | Mutable strategies |
| `results/*.tsv` | Experiment ledger |
| `loop.py`, `backtest.py`, `data_fetch.py`, `program.md` | Harness |
| `app.py`, `web/` | Local dashboard |

---

## dev → main

**Daily at 06:00 PST** (`sync_branches.yml`): `git merge origin/dev` into `main` and push.
No per-file promotion gate — whatever is on `dev` becomes `main`.

`configs.toml` should be edited on `main` (or merged from dev if you keep a copy there).

---

## CI Workflows

| Workflow | Schedule | What it does |
|---|---|---|
| `loop-stocks.yml` | 11:00 UTC (03:00 PST) | Loop on `dev`, stocks |
| `loop-crypto.yml` | 11:00 UTC (03:00 PST) | Loop on `dev`, crypto |
| `sync_branches.yml` | 14:00 UTC (06:00 PST) | Merge `dev` → `main` |
| `paper.yml` | Weekdays / every 4h | `live_trade.py` on `main` |

---

## Key Invariants

1. **The loop only runs on `dev`.**
2. **`main` updates via daily merge** (or manual `git merge dev` on `main`).
3. **`live_trade.py` / paper CI always use `main`.**
4. Avoid local loop runs at 03:00 PST when GHA pushes to `dev`.
