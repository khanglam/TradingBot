# TradingBot — Autonomous Strategy Optimization

## What This Project Is

A **karpathy-autoresearch-style autonomous trading strategy optimizer**. An LLM
agent modifies a strategy file, runs a backtest, measures Sharpe (or Calmar /
DSR), keeps improvements, discards regressions, and loops — exactly like
autoresearch does for LLM training.

A **daily signal scanner** (`scan.py`) deploys the latest strategy.py against
a watchlist and pings a webhook when a buy/sell signal triggers — the
"alert me, I'll execute manually" workflow.

Reference: `karpathy-auto-research/program.md` for the original loop pattern.

## The Stack (chosen 2026-04-28 after `/research`)

| Layer | Choice | Why |
|---|---|---|
| Backtest engine | `backtesting.py` | Pure function call, returns metrics dict, fast |
| Live/paper exec | `lumibot` + Alpaca (optional) | Paper ↔ live via one flag |
| Crypto data | `ccxt` (KuCoin public) | Free, 8+ years OHLCV, no API key |
| Stock data | `yfinance` | Free, daily back to 1990 |
| Storage | Parquet (pyarrow) | 40× faster reads than CSV for backtest loops |
| LLM agent | Claude (Haiku 4.5 default; configurable) via `anthropic` SDK | Drives mutation prompts |
| Scheduler | GitHub Actions cron | Loop daily, scanner weekday mornings |

Jesse was evaluated and archived to `archive/` — too heavy for the tight loop.

## The Autoresearch Loop

```
LOOP FOREVER:
  1. Read current strategy.py + last 10 rows of results.tsv + program.md
  2. Ask Claude for ONE mutation (description + new strategy.py contents)
  3. Write strategy.py, git commit
  4. Compute DSR benchmark from prior trial variance
  5. Run backtest.py with DSR_BENCHMARK in env → parse summary block
  6. Apply keep/discard rules:
       - OPTIMIZE_METRIC > best_so_far AND constraints pass → keep
       - DSR < DSR_GATE_THRESHOLD (if enabled)              → discard
       - max_drawdown ≥ 30% or trades < 20                  → discard
       - else (regression)                                  → discard
       - 0 trades or import error                           → crash (also discard)
       discard / crash → git reset --hard HEAD~1
  7. Append row to results.tsv
  8. Stop after --iters (or human interrupt). Strike-out is opt-in: set `MAX_REGRESSIONS=N` env to freeze after N consecutive non-keeps; off by default.
```

| autoresearch | this project |
|---|---|
| `prepare.py` (fixed) | `backtest.py` — immutable harness |
| `train.py` (mutable) | `strategy.py` — only file the agent edits |
| `val_bpb` (lower=better) | `val_sharpe` / `calmar` / `dsr` (higher=better) |
| `program.md` | `program.md` (constraints + output format) |
| `results.tsv` | `results/results.tsv` |

## Repository Layout

```
TradingBot/
├── CLAUDE.md                  this file
├── README.md                  user-facing docs
├── program.md                 agent constraints + required output format
├── strategy.py                MUTABLE — only file the loop edits
├── backtest.py                FIXED — evaluation harness (single + basket modes)
├── loop.py                    autoresearch orchestrator
├── scan.py                    daily watchlist scanner → webhook alerts
├── live_trade.py              optional Alpaca paper/live executor
├── data_fetch.py              CCXT + yfinance → Parquet
├── app.py                     FastAPI dashboard + SSE
├── web/index.html             dashboard UI (single self-contained file)
├── results/
│   ├── results.tsv            experiment log (committed; survives CI runs)
│   ├── run.log                latest backtest stdout (gitignored)
│   └── scan.log               signal-scan history (gitignored)
├── data/                      cached OHLCV (gitignored)
├── .github/workflows/
│   ├── loop.yml               scheduled autoresearch (daily 02:00 PST)
│   └── scan.yml               scheduled scan (weekdays 05:30 PST)
├── archive/                   old Jesse-based code
└── karpathy-auto-research/    reference, read-only
```

## Backtest Configuration

All settings are env-driven; defaults below. Each `(SYMBOLS × VAL window)` pair
writes to its own `results/<slug>_<years>.tsv` file so changing asset or window
NEVER clobbers prior research. Switching back resumes the right history.

| Env var | Default | Notes |
|---|---|---|
| `SYMBOLS` | `stocks/TSLA_1d,stocks/NVDA_1d,stocks/PYPL_1d` | Comma-sep parquet stems under `data/`. N=1 → single-symbol mode; N≥2 → basket mode. |
| `OPTIMIZE_METRIC` | `val_sharpe` | One of `val_sharpe` / `calmar` / `dsr`. The keep/discard scalar. All metrics logged regardless. |
| `TRAIN_START`/`TRAIN_END` | `2018-01-01` / `2019-12-31` | Agent reasons about; not the metric. |
| `VAL_START`/`VAL_END` | `2020-01-01` / `2024-12-31` | What the loop optimizes. 5y default covers COVID, mania, 2022 bear, 2023 recovery, 2024 bull. |
| `LOCKBOX_START` | `2025-01-01` | Held back; loop never touches. Promote manually. |
| `STARTING_CASH` | `1000000` | Nominal — avoids integer-share rounding. Sharpe & % returns are scale-invariant. |
| `COMMISSION` | `0.0` | 0 for equities; `0.0006` for KuCoin crypto. |
| `MIN_TRADES` | `20` | Single-symbol min; below → val_sharpe forced to 0. |
| `MIN_BASKET_TRADES` | `5` | Per-symbol min in basket mode. |
| `BASKET_PENALTY` | `0.5` | basket score = `mean(sharpe) - PENALTY * std(sharpe)`. |
| `MAX_DRAWDOWN_LIMIT` | `30.0` | Discard threshold (percent). |
| `KEEP_THRESHOLD` | `0.0` | Margin needed above best-so-far to keep. |
| `MAX_REGRESSIONS` | `0` (disabled) | Strike-out brake; karpathy mode by default. |
| `DSR_GATE_THRESHOLD` | `0` (disabled) | Multiple-testing rejection threshold. Enable when N≥50 trials. |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | Agent model. |

Stock data via yfinance (free, daily back to 1990); crypto via ccxt/KuCoin (Binance returns 451 to US IPs).

| Run | Active results file | Baseline (current strategy.py) |
|---|---|---|
| stocks basket (current) | `results/stocks-NVDA-PYPL-TSLA_1d_2020-2024.tsv` | val_sharpe 0.52 · MaxDD 58% · 24 trades · 212% return |
| BTC archive (resume with env) | `results/crypto-BTC_USDT_4h_2023-2024.tsv` | val_sharpe 1.5251 best (achieved iter 8 on commit `6cf72b9`) |

To resume the BTC research:
```bash
SYMBOLS=crypto/BTC_USDT_4h VAL_START=2023-01-01 VAL_END=2024-12-31 \
  COMMISSION=0.0006 .venv/Scripts/python.exe loop.py --iters 20
```
The loop auto-routes to the correct history file.

## Output Format the Loop Parses

`backtest.py` prints:

```
---
val_sharpe:       1.234567
sortino:          1.890123
sharpe_ann_4h:    1.728618
calmar:           1.910234
psr:              0.876543
dsr:              0.654321
skew:             0.123
kurtosis:         5.4
max_drawdown:     12.34
win_rate:         0.456
total_trades:     87
total_return_pct: 45.67
---
```

Crashes / insufficient trades → all zeros, status `crash`.

## results.tsv Schema (current; migrations are in `loop.py:LEGACY_SCHEMAS`)

Tab-separated. Header row + one data row per experiment, in order:

```
commit  val_sharpe  sortino  sharpe_ann_4h  calmar  psr  dsr  skew  kurtosis  max_drawdown  win_rate  total_trades  status  description
```

Schemas evolve append-only. Adding a column? Add it to `RESULTS_COLS` and
extend `LEGACY_SCHEMAS` so old files migrate cleanly.

## Three optimization metrics

| Metric | Picks for | When to use |
|---|---|---|
| `val_sharpe` | Risk-adjusted return | Default. Most defensible. |
| `calmar` | Returns per drawdown | When you want bigger numbers, fewer crashes |
| `dsr` | Statistical significance | After N≥50 trials; corrects for selection bias |

The active metric is the keep/discard scalar; all metrics are logged
regardless. `OPTIMIZE_METRIC=...` env var.

## SYMBOLS — single or basket, one knob

`SYMBOLS` is a comma-separated list of parquet stems under `data/`:

```bash
# Single (MIN_TRADES=20, raw Sharpe)
SYMBOLS=crypto/BTC_USDT_4h .venv/Scripts/python.exe loop.py --iters 20
SYMBOLS=stocks/TSLA_1d     .venv/Scripts/python.exe loop.py --iters 20

# Basket (N≥2; MIN_BASKET_TRADES=5; scored mean(sharpe) - 0.5·std(sharpe))
SYMBOLS=stocks/TSLA_1d,stocks/NVDA_1d,stocks/AAPL_1d,stocks/MSFT_1d \
  .venv/Scripts/python.exe loop.py --iters 20
```

Resolves to `data/{SYMBOLS[i]}.parquet`. Fetch with `data_fetch.py` first.
Mode is auto-selected by count — no separate basket flag. Empty/unset →
`crypto/BTC_USDT_4h`. Tune the basket overfit penalty via `BASKET_PENALTY`
(default 0.5).

## Scheduled execution (GitHub Actions)

| Workflow | Trigger | Purpose |
|---|---|---|
| `.github/workflows/loop.yml` | Daily 02:00 PST + dispatch | Loop runs, commits keeps + results.tsv back to main |
| `.github/workflows/scan.yml` | Weekdays 05:30 PST + dispatch | Scanner runs, posts to webhook |

Required repo secrets / variables:

| Name | Type | Used by | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Secret | loop.yml | Claude API |
| `SCAN_WEBHOOK_URL` | Secret | scan.yml | Discord/Slack webhook |
| `CLAUDE_MODEL` | Variable | loop.yml | Override model |
| `SCAN_WATCHLIST` | Variable | scan.yml | Comma-sep symbols |

## Dashboard UI

Two files: `app.py` (FastAPI server + SSE for live loop output) and
`web/index.html` (the entire UI — Tailwind + Alpine + Chart.js + Prism, all
loaded from CDNs). No build step.

```bash
.venv/Scripts/python.exe app.py
```

Then open http://127.0.0.1:8000.

API endpoints (under `/api/*`): `summary`, `results`, `strategy`, `program`,
`equity`, `git-log`, `setup`, `backtest`, `loop/start`, `loop/stop`,
`loop/status`, `loop/stream`.

## Hard Rules for the Agent

These are mirrored in `program.md` and enforced by `loop.py`:

- **CAN edit**: `strategy.py` only
- **CANNOT edit**: `backtest.py`, `loop.py`, `data_fetch.py`, `live_trade.py`,
  `scan.py`, the windows, the harness internals
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

## Status (2026-04-29)

- [x] Stack chosen via `/research`
- [x] All core scripts written: backtest, loop, scan, live_trade, app
- [x] Old Jesse code archived to `archive/`
- [x] Folder reorg: results/, web/, .github/workflows/
- [x] Calmar / Sortino / PSR / DSR / skew / kurtosis logged per trial
- [x] Lockbox window (2025+) reserved
- [x] DSR multiple-testing gate plumbed (off by default until N≥50)
- [x] Basket-mode optimization for stocks
- [x] Daily signal scanner + Discord/Slack webhook
- [x] Scheduled GitHub Actions for both loop and scanner
- [ ] First scheduled CI run executed
- [ ] First scan signal received
- [ ] First lockbox-window evaluation
