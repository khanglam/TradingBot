# Agent Guidelines — TradingBot

This file is the authoritative reference for repository structure and branch ownership.
**Rule: any time a file is added, moved, renamed, or deleted on any branch, update this file
to reflect the change before committing.**

---

## Repository Purpose

Karpathy-style autoresearch optimizer for trading strategies. The loop mutates a strategy
file, runs a backtest, and keeps or discards the change. Good strategies are promoted to
`main` where they run in live paper trading.

---

## Two-Branch Structure

### `main` — deployment surface & frozen strategies

Never mutated by the loop. Updated by `sync_branches.py` (strategy promotion) and
manual commits. `paper.yml` and `live_trade.py` always read from here.

| File / Dir | Purpose |
|---|---|
| `live_trade.py` | Alpaca paper/live executor |
| `sync_branches.py` | Promotion gate: dev candidate → main if it beats frozen |
| `configs.toml` | Campaign config — **lives only on main**; dev reads via `git show origin/main:configs.toml` |
| `strategies/crypto.py` | **Frozen** crypto strategy — promoted from `dev` |
| `strategies/stocks.py` | **Frozen** stocks strategy — promoted from `dev` |
| `.github/workflows/` | All CI workflow definitions — GitHub reads these from main |
| `README.md`, `.gitignore`, `.env.example` | Docs and local config |

Harness files (`loop.py`, `backtest.py`, etc.) may exist on `main` for promotion
backtests; primary development happens on `dev`. Merge `dev` → `main` when harness
should stay in sync.

### `dev` — research branch

All loop mutations, experiment logs, and harness edits. The loop runs here only;
bad experiments are `git reset --hard`'d away.

| File / Dir | Purpose |
|---|---|
| `strategies/crypto.py` | **Mutable** crypto candidate |
| `strategies/stocks.py` | **Mutable** stocks candidate |
| `results/*.tsv` | Per-campaign experiment ledger |
| `loop.py`, `backtest.py`, `data_fetch.py`, `program.md` | Harness — edit here |
| `app.py`, `web/` | Local dashboard (dev checkout) |

**Does NOT deploy to paper:** strategies on `dev` are candidates until promoted.

---

## Sync Rules

### dev → main (promotion, daily via `sync_branches.yml`)

Only `strategies/<campaign>.py` is copied from `dev` to `main`, and only if the
candidate beats the frozen strategy on the val window AND clears lockbox sanity floors
(`PROMOTION_MARGIN`, `LOCKBOX_MIN_SHARPE`, `LOCKBOX_MAX_DD`).

### configs.toml — main only

`configs.toml` lives exclusively on `main`. `dev` CI and local tools read it with:

```python
subprocess.run(["git", "show", "origin/main:configs.toml"], ...)
```

### Harness — manual merge

There is no automated harness sync. After changing `loop.py` / `backtest.py` on `dev`,
merge to `main` when promotion should use the same harness version.

---

## CI Workflows

| Workflow | Triggers | Checks out | What it does |
|---|---|---|---|
| `loop-stocks.yml` | Daily 11:00 UTC (03:00 PST) | `dev` | 10 loop iters for stocks campaign |
| `loop-crypto.yml` | Daily 11:00 UTC (03:00 PST) | `dev` | 10 loop iters for crypto campaign |
| `loop-dev.yml` | `workflow_call` | `dev` | Reusable loop job |
| `sync_branches.yml` | Daily 12:00 UTC | `main` (+ fetch `dev`) | Promotion gauntlet; push main if passed |
| `paper.yml` | Stocks weekdays 13:35 UTC; crypto every 4h | `main` | `live_trade.py` against frozen strategies |

---

## Key Invariants

1. **The loop never touches main.** All mutations stay on `dev` until `sync_branches.py` promotes them.
2. **Avoid local loop runs at 03:00 PST** when scheduled GHA loops push to `dev`.
3. **`strategies/<campaign>.py` on main is always the last promoted (frozen) version.**
   `live_trade.py` always reads from main.
4. **`backtest.py` is immutable once stable** within a research session. Treat harness changes as deliberate.
5. **`configs.toml` changes on main take effect immediately** for CI via `origin/main` at runtime.
