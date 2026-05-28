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
| `live_trade.py` | Alpaca paper/live executor. Refuses to start unless HEAD has a PASS row in `results/promotions.tsv` (bypass with `ALLOW_UNPROMOTED=1`). |
| `promote.py` | One-shot lockbox gate. Run manually before paper trading; appends to `results/promotions.tsv`. |
| `configs.toml` | Campaign config for crypto/stocks (`CAMPAIGN` env); read from the checked-out branch |
| `strategies/*.py` | What paper trades — copy of dev after daily merge |
| `.github/workflows/` | CI definitions (read from default branch) |

### `dev` — research branch

All loop work happens here. The loop optimizes on **train** and gates on
**val** (with regime-stability sub-period checks); **lockbox** is the third
holdout, consumed only by `promote.py`.

| File / Dir | Purpose |
|---|---|
| `strategies/*.py` | Mutable strategies (declare `MIN_BARS_REQUIRED` for live fetch sizing) |
| `results/*.tsv` | Experiment ledger (22-col schema; `promotions.tsv` is the lockbox audit) |
| `loop.py`, `backtest.py`, `data_fetch.py`, `program.md` | Harness |
| `app.py`, `web/` | Local dashboard |

---

## dev → main

**Daily at 06:00 California** (`sync_branches.yml`): `git merge origin/dev` into `main` and push.
No per-file promotion gate — whatever is on `dev` becomes `main`.

`configs.toml` on `dev` is what the loop and dashboard use; merge `dev` → `main` daily so paper/CI match.

---

## CI Workflows

| Workflow | Schedule | What it does |
|---|---|---|
| `loop-stocks.yml` | 03:00 `America/Los_Angeles` | Loop on `dev`, stocks (disable workflow in Actions to pause) |
| `loop-crypto.yml` | 03:00 `America/Los_Angeles` | Loop on `dev`, crypto (disable workflow in Actions to pause) |
| `sync_branches.yml` | 06:00 `America/Los_Angeles` | Merge `dev` → `main` |
| `paper.yml` | Weekdays / every 4h | `live_trade.py` on `main` |

---

## Key Invariants

1. **The loop only runs on `dev`.**
2. **`main` updates via daily merge** (or manual `git merge dev` on `main`).
3. **`live_trade.py` / paper CI always use `main`.**
4. Avoid local loop runs at 03:00 California when GHA pushes to `dev`.
