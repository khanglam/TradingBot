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

## Three-Branch Structure

### `main` — deployment surface & frozen strategies

The only branch humans and CI workflows interact with directly for deployment.
Never mutated by the loop. Only updated by `sync_branches.py` (strategy promotion)
and manual commits.

| File / Dir | Purpose |
|---|---|
| `app.py` | FastAPI dashboard server (local dev only) |
| `web/index.html` | Dashboard UI served by app.py |
| `scan.py` | Daily signal scanner — emits BUY/SELL to webhook |
| `live_trade.py` | Alpaca paper/live executor |
| `sync_branches.py` | Promotion gate: candidate → main if it beats frozen |
| `loop.py` | Autoresearch orchestrator (also synced to campaign branches) |
| `backtest.py` | Evaluation harness (also synced to campaign branches) |
| `data_fetch.py` | OHLCV downloader (also synced to campaign branches) |
| `program.md` | LLM system prompt for the loop (also synced) |
| `requirements.txt` | Python dependencies (also synced) |
| `AGENTS.md` | This file (also synced) |
| `configs.toml` | Campaign config — **lives only on main**; campaign branches read it via `git show origin/main:configs.toml` |
| `strategies/crypto.py` | **Frozen** crypto strategy — promoted from `autoresearch/crypto` |
| `strategies/stocks.py` | **Frozen** stocks strategy — promoted from `autoresearch/stocks` |
| `.github/workflows/` | All CI workflow definitions — GitHub reads these from main |
| `.claude/` | Claude Code skills and local settings |
| `README.md`, `.gitignore`, `.env.example`, `.cursorrules` | Docs and local config |

### `autoresearch/crypto` — crypto research loop

The loop runs here. `strategies/crypto.py` is mutated every iteration; bad experiments
are `git reset --hard`'d away. Only `strategies/crypto.py` and `results/*.tsv` change.
Everything else arrives via harness sync from main.

| File / Dir | Purpose |
|---|---|
| `strategies/crypto.py` | **Mutable** — the strategy being evolved |
| `results/crypto-*.tsv` | Experiment ledger (keep/discard rows appended by loop) |
| `results/.gitkeep` | Keeps the results/ dir in git |
| `loop.py`, `backtest.py`, `data_fetch.py` | Harness — synced from main, do not edit here |
| `program.md`, `requirements.txt`, `AGENTS.md` | Harness — synced from main |
| `.gitignore`, `.env.example`, `.cursorrules`, `README.md` | Config/docs |

**Does NOT contain:** `app.py`, `scan.py`, `live_trade.py`, `sync_branches.py`, `web/`,
`.github/`, `.claude/`, `archive/`, `strategies/stocks.py`, `configs.toml`.

### `autoresearch/stocks` — stocks research loop

Mirror of `autoresearch/crypto` for the stocks campaign.

| File / Dir | Purpose |
|---|---|
| `strategies/stocks.py` | **Mutable** — the strategy being evolved |
| `results/stocks-*.tsv` | Experiment ledger |
| Everything else | Same as crypto branch above |

**Does NOT contain:** same exclusions as crypto branch, plus `strategies/crypto.py`.

---

## Sync Rules

### main → campaign (harness sync, daily via `sync_branches.yml`)
Only these files are copied from main to campaign branches:
```
loop.py  backtest.py  program.md  data_fetch.py  requirements.txt  AGENTS.md
```
**Do not add** `app.py`, `scan.py`, `live_trade.py`, `sync_branches.py`, `web/`,
`.github/`, or `configs.toml` to this list.

### campaign → main (promotion, daily via `sync_branches.yml`)
Only `strategies/<campaign>.py` is copied from a campaign branch to main, and only
if the candidate beats the frozen strategy on the val window AND clears the lockbox
sanity floors (configurable via `PROMOTION_MARGIN`, `LOCKBOX_MIN_SHARPE`, `LOCKBOX_MAX_DD`).

### configs.toml — main only
`configs.toml` lives exclusively on `main`. Campaign branches and CI read it with:
```python
subprocess.run(["git", "show", "origin/main:configs.toml"], ...)
```
Never commit `configs.toml` to a campaign branch. Never add it to the harness sync list.

---

## CI Workflows

| Workflow | Triggers | Checks out | What it does |
|---|---|---|---|
| `loop-stocks.yml` | Daily 04:00 UTC | — | Calls `loop-campaign.yml` with `campaign=stocks` |
| `loop-crypto.yml` | Every 6h | — | Calls `loop-campaign.yml` with `campaign=crypto` |
| `loop-campaign.yml` | `workflow_call` | `autoresearch/<campaign>` | Fetches `origin/main` for config, runs `loop.py`, pushes commits back |
| `sync_branches.yml` | Daily 12:00 UTC | `main` (+ fetches campaign) | Runs `sync_branches.py`; if promoted, commits to main; syncs harness to campaign |
| `scan.yml` | Stocks 13:30 UTC weekdays; crypto every 4h | `main` | Runs `scan.py`, posts signals to webhook |
| `paper.yml` | Stocks 13:35 UTC weekdays; crypto every 4h | `main` | Runs `live_trade.py`, submits Alpaca orders |

---

## Key Invariants

1. **The loop never touches main.** All mutations stay on the campaign branch until
   `sync_branches.py` explicitly promotes them.
2. **Campaign branches are single-writer.** Only the loop workflow commits to them.
   Do not push directly to `autoresearch/crypto` or `autoresearch/stocks`.
3. **`strategies/<campaign>.py` on main is always the last promoted (frozen) version.**
   `scan.py` and `live_trade.py` always read from main.
4. **`backtest.py` is immutable once stable.** Changing it mid-experiment makes runs
   incomparable. Treat it as a fixed harness.
5. **`configs.toml` changes take effect immediately** — CI reads from `origin/main` at
   runtime, so a pushed config change is live on the next loop run without any branch sync.
