# TradingBot — Autonomous Strategy Optimization

## What This Project Is

A **karpathy-autoresearch-style autonomous trading strategy optimizer**. An LLM
agent modifies a strategy file, runs a backtest, measures Sharpe (or Calmar /
DSR), keeps improvements, discards regressions, and loops — exactly like
autoresearch does for LLM training.

A **daily signal scanner** (`scan.py`) deploys the latest strategy.py against
a watchlist and pings a webhook when a buy/sell signal triggers — the
"alert me, I'll execute manually" workflow.

Reference: https://github.com/karpathy/autoresearch for the original loop pattern.

## Branching model (the load-bearing design choice)

The project diverges from karpathy in one critical way: there's a live
consumer of the strategy file (`scan.py`, `live_trade.py`, paper trading)
that must NOT see mid-mutation state. We use branches to enforce that:

```
main                    ← FROZEN. scan.py + live_trade.py read this.
                          Updated only by promote.py (daily, validated).
  ↑ (promotion gate, single-file copy)
autoresearch/stocks     ← stocks loop owns this. Mutates strategies/stocks.py
                          + writes results/<slug>.tsv. Single writer
                          (loop.yml). git reset --hard happens here.
autoresearch/crypto     ← same shape, for crypto.
```

Rules:
- The loop NEVER commits to main. It refuses to start if HEAD is on main
  (override: `ALLOW_LOOP_ON_MAIN=1`).
- Each campaign branch has exactly one writer (its CI workflow), so there
  are no push races and no rebase-retry dance.
- `promote.py` is the only path between candidate and frozen. It runs a
  validation gauntlet (val window must beat frozen by margin; lockbox
  must clear sanity floors) before overwriting the frozen file on main.
- Local research: `git checkout autoresearch/stocks` (or `crypto`) before
  running `python loop.py`. Locally promote via `python promote.py --campaign stocks`.

## The Stack (chosen 2026-04-28 after `/research`)

| Layer | Choice | Why |
|---|---|---|
| Backtest engine | `backtesting.py` | Pure function call, returns metrics dict, fast |
| Paper/live exec | `alpaca-py` (Alpaca paper) | Stocks + crypto in one account; the executor in `live_trade.py` reuses the same `backtesting.py` harness `scan.py` uses, so it always tracks the latest `strategy.py` |
| Crypto data (loop) | `ccxt` (KuCoin public) | Free, 8+ years OHLCV, no API key |
| Crypto data (paper) | `alpaca-py` `CryptoHistoricalDataClient` | Same broker as execution; one less dependency |
| Stock data | `yfinance` | Free, daily back to 1990 |
| Storage | Parquet (pyarrow) | 40× faster reads than CSV for backtest loops |
| LLM agent | OpenRouter via `openai` SDK; default `anthropic/claude-haiku-4-5`, any model slug works | Drives mutation prompts |
| Scheduler | GitHub Actions cron | Stocks loop daily overnight, crypto loop every 6h, daily promotion gate, scanner + paper on their own cadences |

Jesse was evaluated and archived to `archive/` — too heavy for the tight loop.
Lumibot was tried and dropped — its strategy class hard-coded an EMA crossover
that diverged from `strategy.py` whenever the autoresearch loop changed it.
The `alpaca-py`-based executor in `live_trade.py` reuses the in-tree `Strategy`
class and the same `backtesting.Backtest` harness as `scan.py`, so the paper
trader automatically inherits whatever the loop optimized last.

## The Autoresearch Loop

```
LOOP FOREVER:
  1. Read current STRATEGY_FILE (per-campaign) + last 10 rows of tsv + program.md
  2. Ask Claude for ONE mutation (description + new strategy file contents)
  3. Write strategy file, git commit
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
| `train.py` (mutable) | `strategies/<campaign>.py` — only file the agent edits per campaign |
| `val_bpb` (lower=better) | `val_sharpe` / `calmar` / `dsr` (higher=better) |
| `program.md` | `program.md` (constraints + output format) |
| `results.tsv` | `results/results.tsv` |

## Repository Layout

```
TradingBot/
├── CLAUDE.md                  this file
├── README.md                  user-facing docs
├── program.md                 agent constraints + required output format
├── strategies/
│   ├── stocks.py              MUTABLE — stocks campaign edits this only
│   └── crypto.py              MUTABLE — crypto campaign edits this only
├── backtest.py                FIXED — evaluation harness (single + basket modes)
├── loop.py                    autoresearch orchestrator
├── scan.py                    daily watchlist scanner → webhook alerts
├── live_trade.py              Alpaca paper/live executor (reuses scan.py logic)
├── data_fetch.py              CCXT + yfinance → Parquet
├── app.py                     FastAPI dashboard + SSE
├── web/index.html             dashboard UI (single self-contained file)
├── results/
│   ├── <slug>_<years>.tsv     per-campaign experiment log (committed)
│   ├── run.log                latest backtest stdout (gitignored)
│   ├── scan.log               signal-scan history (gitignored)
│   └── paper.log              paper-trade execution history (gitignored)
├── data/                      cached OHLCV (gitignored)
├── .github/workflows/
│   ├── loop.yml               autoresearch — stocks daily 04:00 UTC, crypto every 6h
│   ├── promote.yml            daily promotion gate (12:00 UTC)
│   ├── scan.yml               stocks pre-market + crypto every 4h
│   └── paper.yml              Alpaca paper-trade executor (mirrors scan)
├── promote.py                 candidate → frozen validation gate
└── archive/                   old Jesse + Lumibot code
```

## Backtest Configuration

All settings are env-driven; defaults below. Each `(SYMBOLS × VAL window)` pair
writes to its own `results/<slug>_<years>.tsv` file so changing asset or window
NEVER clobbers prior research. Switching back resumes the right history.

| Env var | Default | Notes |
|---|---|---|
| `SYMBOLS` | `stocks/TSLA_1d,stocks/NVDA_1d,stocks/PYPL_1d` | Comma-sep parquet stems under `data/`. N=1 → single-symbol mode; N≥2 → basket mode. |
| `STRATEGY_FILE` | derived from SYMBOLS prefix | `strategies/stocks.py` or `strategies/crypto.py`. Each campaign owns its own file so stocks and crypto evolve independently. |
| `OPTIMIZE_METRIC` | `val_sharpe` | One of `val_sharpe` / `calmar` / `dsr`. The keep/discard scalar. All metrics logged regardless. |
| `TRAIN_START`/`TRAIN_END` | `2018-01-01` / `2019-12-31` | Agent reasons about; not the metric. |
| `VAL_START`/`VAL_END` | `2020-01-01` / `2024-12-31` | What the loop optimizes. 5y default covers COVID, mania, 2022 bear, 2023 recovery, 2024 bull. |
| `LOCKBOX_START` | `2025-01-01` | Held back; loop never touches. Promote manually. |
| `STARTING_CASH` | `1000000` | Nominal — avoids integer-share rounding. Sharpe & % returns are scale-invariant. |
| `COMMISSION` | `0.001` | Per side (so 10 bps round-trip). Slippage+spread proxy that kills micro-scalp overfits. Same value for stocks and crypto by default. |
| `MIN_TRADES` | `20` | Single-symbol min; below → val_sharpe forced to 0. |
| `MIN_BASKET_TRADES` | `5` | Per-symbol min in basket mode. |
| `BASKET_PENALTY` | `0.5` | basket score = `mean(sharpe) - PENALTY * std(sharpe)`. |
| `MAX_DRAWDOWN_LIMIT` | `30.0` | Discard threshold (percent). |
| `KEEP_THRESHOLD` | `0.0` | Margin needed above best-so-far to keep. |
| `MAX_REGRESSIONS` | `0` (disabled) | Strike-out brake; karpathy mode by default. |
| `DSR_GATE_THRESHOLD` | `0` (disabled) | Multiple-testing rejection threshold. Enable when N≥50 trials. |
| `OPENROUTER_MODEL` | `anthropic/claude-haiku-4-5` | Any model slug from openrouter.ai/models — Anthropic, OpenAI, DeepSeek, Google, etc. |

Stock data via yfinance (free, daily back to 1990); crypto via ccxt/KuCoin (Binance returns 451 to US IPs).

| Run | Active results file | Baseline (current strategy.py) |
|---|---|---|
| stocks basket (current) | `results/stocks-NVDA-PYPL-TSLA_1d_2020-2024.tsv` | val_sharpe 0.52 · MaxDD 58% · 24 trades · 212% return |
| BTC archive (resume with env) | `results/crypto-BTC_USDT_4h_2023-2024.tsv` | val_sharpe 1.5251 best (achieved iter 8 on commit `6cf72b9`) |

To resume the BTC research:
```bash
SYMBOLS=crypto/BTC_USDT_4h VAL_START=2023-01-01 VAL_END=2024-12-31 \
  COMMISSION=0.001 .venv/Scripts/python.exe loop.py --iters 20
```
The loop auto-routes to the correct history file.

**Commission was bumped from 0.0/0.0006 → 0.001 on 2026-04-30.** Prior
experiments under the cheaper costs are not directly comparable; the loop
will re-converge from scratch under the new constraint. Old tsv rows are
kept for historical reference but `best_metric_so_far()` will be reset
in practice as the new realistic-cost world rejects most prior bests.

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
| `.github/workflows/loop.yml` | Stocks: 04:00 UTC daily (= 21:00 PST prior day, well outside US market hours). Crypto: every 6h (`0 */6 * * *`). Each campaign runs on its own branch (`autoresearch/<campaign>`) — single writer per branch, no push races. | Two parallel autoresearch tracks; commits land on the campaign branch only — never on main |
| `.github/workflows/promote.yml` | Daily 12:00 UTC (~04:00 PST / 05:00 PDT, before stocks paper at 13:35 UTC). For each campaign, fetches `autoresearch/<campaign>`, runs `promote.py` (val-window beat + lockbox sanity floors), and if it passes, commits the new frozen `strategies/<campaign>.py` to main. | The only path from candidate → frozen |
| `.github/workflows/scan.yml` | Stocks: weekdays 13:30 UTC (05:30 PST). Crypto: every 4h, every day (`0 */4 * * *`). Reads `strategies/<campaign>.py` from main (frozen). | Signal alerts to webhook (Discord/Slack format) |
| `.github/workflows/paper.yml` | Stocks: weekdays 13:35 UTC (5 min after scan). Crypto: every 4h. Reuses `scan.py`'s logic via `live_trade.py`; reads frozen strategies from main. | Paper-trade the latest promoted strategy against the watchlists |

Required repo secrets / variables:

| Name | Type | Used by | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Secret | loop.yml | OpenRouter API key (https://openrouter.ai/keys) |
| `SCAN_WEBHOOK_URL` | Secret | scan.yml | Discord/Slack webhook |
| `ALPACA_API_KEY` | Secret | paper.yml | Alpaca paper API key |
| `ALPACA_API_SECRET` | Secret | paper.yml | Alpaca paper API secret |
| `OPENROUTER_MODEL` | Variable | loop.yml | Override default model (any openrouter.ai slug) |
| `SCAN_WATCHLIST` | Variable | scan.yml | Stock watchlist (e.g. `SPY,QQQ,TSLA,NVDA`) |
| `SCAN_WATCHLIST_CRYPTO` | Variable | scan.yml | Crypto watchlist via yfinance (e.g. `BTC-USD,ETH-USD`) |
| `PAPER_WATCHLIST` | Variable | paper.yml | Stock watchlist (Alpaca tickers) |
| `PAPER_CRYPTO_WATCHLIST` | Variable | paper.yml | Crypto watchlist (Alpaca pairs, e.g. `BTC/USD,ETH/USD`) |
| `PAPER_PER_SYMBOL_CASH` | Variable | paper.yml | Per-symbol cash budget (default `10000`) |

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

- **CAN edit**: the active campaign strategy file only (`strategies/stocks.py` or `strategies/crypto.py` — never both, never anything else)
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
- [x] Scheduled GitHub Actions for loop, scanner, and paper trader
- [x] Branch-per-campaign: stocks/crypto loops isolated on `autoresearch/*` branches, main reserved for promoted strategies
- [x] Promotion gate (`promote.py` + `promote.yml`): val-window beat + lockbox sanity floors before main moves
- [x] Paper trading (`live_trade.py`) rewritten on `alpaca-py` — reuses `scan.py`'s strategy harness so it always tracks the latest `strategy.py`
- [x] $10K per-symbol cash default, idempotent re-runs, JSONL `paper.log`
- [ ] User signs up for Alpaca paper keys + sets `ALPACA_API_KEY`/`ALPACA_API_SECRET` repo secrets
- [ ] First scheduled CI run executed (loop + scan + paper)
- [ ] First paper order filled
- [ ] First lockbox-window evaluation
