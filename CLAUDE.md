# TradingBot — Autonomous Strategy Optimization

## What This Project Is

A **karpathy-autoresearch-style autonomous trading strategy optimizer**. An LLM
agent modifies a strategy file, runs a backtest, measures Sharpe, keeps
improvements, discards regressions, and loops — exactly like autoresearch does
for LLM training.

Reference: `karpathy-auto-research/program.md` for the original loop pattern.

## The Stack (chosen 2026-04-28 after `/research`)

| Layer | Choice | Why |
|---|---|---|
| Backtest engine | `backtesting.py` | Pure function call, returns metrics dict, no CLI, fast |
| Live/paper exec | `lumibot` + Alpaca | One config flag flips paper ↔ live, supports stocks + crypto |
| Crypto data | `ccxt` (Binance public) | Free, 8+ years OHLCV, no API key |
| Stock data | `yfinance` | Free, daily back to 1990 |
| Storage | Parquet (pyarrow) | 40× faster reads than CSV for backtest loops |
| LLM agent | Claude Sonnet 4.6 via `anthropic` SDK | Drives the mutation prompts |

Jesse was evaluated and archived to `archive/` — too heavy for the tight loop.

## The Autoresearch Loop

```
LOOP FOREVER:
  1. Read current strategy.py + last 10 rows of results.tsv + program.md
  2. Ask Claude for ONE mutation (description + new strategy.py contents)
  3. Write strategy.py, git commit
  4. Run backtest.py → parses summary block from stdout
  5. Apply keep/discard rules:
       - val_sharpe > best_so_far AND constraints pass → keep
       - else                                          → git reset --hard HEAD~1
  6. Append row to results.tsv
  7. Stop after --iters or 3 consecutive regressions
```

| autoresearch | this project |
|---|---|
| `prepare.py` (fixed) | `backtest.py` — immutable harness |
| `train.py` (mutable) | `strategy.py` — only file the agent edits |
| `val_bpb` (lower=better) | `val_sharpe` (higher=better) |
| `program.md` | `program.md` (constraints + output format) |
| `results.tsv` | `results.tsv` |

## Repository Layout

```
TradingBot/
├── CLAUDE.md                  this file
├── program.md                 agent constraints + required output format
├── backtest.py                FIXED — evaluation harness, never edit after stable
├── strategy.py                MUTABLE — only file the loop edits
├── loop.py                    autoresearch orchestrator
├── live_trade.py              paper / live execution via LumiBot
├── data_fetch.py              CCXT + yfinance -> Parquet
├── ui/
│   ├── server.py              FastAPI backend + SSE for live loop output
│   └── static/
│       ├── index.html         single-page dashboard (Tailwind + Alpine)
│       ├── app.js             Alpine + Chart.js front-end logic
│       └── styles.css         small custom layer over Tailwind
├── run_ui.bat / run_ui.sh     one-click launcher (uvicorn on :8000)
├── data/                      cached OHLCV (gitignored)
├── results.tsv                experiment log (gitignored)
├── run.log                    latest backtest stdout (gitignored)
├── archive/                   old Jesse-based code
└── karpathy-auto-research/    reference, read-only
```

## Backtest Configuration (FIXED — changing breaks comparability)

| Parameter | Value |
|---|---|
| Asset | BTC/USDT |
| Source | KuCoin via CCXT (Binance returns 451 to US IPs) |
| Timeframe | 4h |
| Train window | 2019-01-01 → 2022-12-31 |
| Validation window | 2023-01-01 → 2024-12-31 |
| Nominal cash | $1,000,000 (large to avoid backtesting.py integer-share rounding; metrics are scale-invariant) |
| Commission | 0.06% (KuCoin taker) |
| Leverage | 1x |
| Min trades | 20 (below → val_sharpe forced to 0) |
| Baseline (EMA 20/50) | Sharpe 0.96 · MaxDD 28.7% · Trades 37 · Return 125% (validation) |

## Output Format the Loop Parses

`backtest.py` prints:

```
---
val_sharpe:       1.234567
sortino:          1.890123
max_drawdown:     12.34
win_rate:         0.456
total_trades:     87
total_return_pct: 45.67
---
```

Crashes / insufficient trades → all zeros, status `crash`.

## results.tsv Schema

Tab-separated, gitignored, append-only:

```
commit  val_sharpe  max_drawdown  win_rate  total_trades  status  description
a1b2c3d 1.234567    12.30         0.480     87            keep    baseline EMA 20/50
b2c3d4e 1.456789    10.10         0.512     94            keep    added RSI<30 filter
c3d4e5f 0.987654    18.20         0.401     61            discard tried Bollinger exits
d4e5f6g 0.000000    0.00          0.000     0             crash   syntax error in init
```

## Setup

```bash
# 1. install deps (first time)
.venv/Scripts/python.exe -m pip install backtesting lumibot ccxt yfinance anthropic pyarrow

# 2. download data
.venv/Scripts/python.exe data_fetch.py --asset crypto --symbol BTC/USDT --timeframe 4h --start 2019-01-01

# 3. set keys in .env
#    ANTHROPIC_API_KEY=...
#    ALPACA_API_KEY=...           (only needed for live_trade.py)
#    ALPACA_API_SECRET=...
#    ALPACA_PAPER=True

# 4. baseline backtest
.venv/Scripts/python.exe backtest.py

# 5. start the loop
.venv/Scripts/python.exe loop.py --iters 50

# 6. paper trade the current strategy.py
.venv/Scripts/python.exe live_trade.py --symbol SPY --asset stock
```

## Dashboard UI

Custom FastAPI + HTML/Tailwind/Alpine/Chart.js dashboard. No build step,
no npm — Tailwind + Alpine + Chart.js + Prism are loaded from CDNs.

```bash
# Windows: double-click run_ui.bat
.venv/Scripts/python.exe -m uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

Then open http://localhost:8000. Single-page layout:

- **Header bar** — best Sharpe · loop status pill · launch/stop button
- **KPI cards** — best sharpe, total experiments, keep rate, current status
- **Equity curve + drawdown** — Chart.js, strategy vs Buy & Hold on validation window
- **Sharpe progression** — best-so-far line across kept experiments
- **Experiments table** — sortable, filterable, color-coded by status
- **Strategy panel** — `strategy.py` syntax-highlighted (Prism) + git history sidebar
- **Setup section** — env check, data file inventory
- **Live console** — SSE stream of loop stdout, slides up from the bottom when the loop runs

API endpoints (under `/api/*`): `summary`, `results`, `strategy`, `program`,
`equity`, `git-log`, `setup`, `backtest`, `loop/start`, `loop/stop`,
`loop/status`, `loop/stream`. Loop runs as a managed subprocess inside
the FastAPI process — start it from the UI and watch live, or stop it at
any time.

## Hard Rules for the Agent

These are mirrored in `program.md` and enforced by `loop.py`:

- **CAN edit**: `strategy.py` only
- **CANNOT edit**: `backtest.py`, `loop.py`, `data_fetch.py`, `live_trade.py`,
  the windows, jesse internals, lumibot internals
- **One change per experiment** (no bundled mutations)
- **No look-ahead** (only bars `[0..current]`)
- **Class signature fixed**: `class Strategy(backtesting.Strategy)` with
  `init` and `next` methods
- **Simpler is better** — equal Sharpe with less code wins

## Skills Available

- `/research <topic>` — fan-out parallel research (5 agents, different angles)
- `/debate <claim>` — stochastic multi-agent consensus (3 agents, adversarial)
- `/scout <topic>` — research → debate pipeline in one command

Use `/scout` before adopting any new framework or major architectural change.

## Status (2026-04-28)

- [x] Stack chosen via `/research`
- [x] `backtest.py`, `strategy.py`, `loop.py`, `live_trade.py`, `data_fetch.py`, `program.md` written
- [x] Old Jesse code archived to `archive/`
- [ ] Dependencies installed (in progress)
- [ ] Initial data download
- [ ] Baseline backtest executed
- [ ] First loop iteration verified
