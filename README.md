# TradingBot

A self-improving trading strategy. An LLM proposes mutations to `strategy.py`,
the harness runs a backtest, and only improvements are kept. Loops forever.

It also ships a daily **signal scanner** (`scan.py`) that runs the current
strategy against a watchlist and pings a webhook when a buy/sell signal
fires — the workflow for "alert me, I'll execute manually."

## Layout

```
TradingBot/
├── strategy.py             ← MUTABLE — the only file the agent edits
├── backtest.py             ← FIXED — evaluation harness; do not edit after baseline
├── loop.py                 ← Orchestrator: ask Claude → backtest → keep/discard
├── scan.py                 ← Daily watchlist scanner; emits BUY/SELL alerts to webhook
├── live_trade.py           ← Alpaca paper/live executor (alpaca-py; reuses scan.py's harness)
├── data_fetch.py           ← OHLCV downloader (CCXT for crypto, yfinance for stocks)
├── app.py                  ← FastAPI dashboard server
├── program.md              ← System prompt: rules + output format for the LLM
├── web/index.html          ← Dashboard UI (single self-contained file)
├── results/                ← Runtime outputs
│   ├── <slug>_<years>.tsv  ← Per-campaign experiment log (committed)
│   ├── run.log             ← Latest backtest stdout (gitignored)
│   ├── scan.log            ← Append-only signal-scan log (gitignored)
│   └── paper.log           ← Paper-trade execution history (gitignored)
├── data/                   ← Cached parquet OHLCV (gitignored)
├── archive/                ← Old code retained for reference
└── .github/workflows/
    ├── loop.yml            ← Autoresearch every 6h, matrix(stocks, crypto)
    ├── scan.yml            ← Stocks pre-market + crypto every 4h
    └── paper.yml           ← Alpaca paper executor (mirrors scan)
```

## First-time setup

```bash
# 1. Install Python dependencies
.venv/Scripts/python.exe -m pip install \
    backtesting ccxt yfinance openai pyarrow scipy python-dotenv \
    fastapi "uvicorn[standard]" sse-starlette

# 2. Download historical data (4h BTC for the default loop)
.venv/Scripts/python.exe data_fetch.py --asset crypto --symbol BTC/USDT --timeframe 4h --start 2019-01-01

# 3. Copy .env.example → .env and fill in OPENROUTER_API_KEY
#    Get a key at https://openrouter.ai/keys (one key, any model)
cp .env.example .env
# Then edit .env

# 4. Make sure the git tree is clean
git add -A && git commit -m "scaffold"
```

## Run the dashboard

```bash
.venv/Scripts/python.exe app.py
# Open http://127.0.0.1:8000
```

KPIs, equity curve, Sharpe progression, experiment table, current
`strategy.py`, git history, and a launch button for the loop with a
live console.

## Running the autoresearch loop

```bash
# Default — backtesting.py reported Sharpe (risk-adjusted return)
.venv/Scripts/python.exe loop.py --iters 10

# Alternative — return-aware Calmar (return / max_drawdown)
OPTIMIZE_METRIC=calmar .venv/Scripts/python.exe loop.py --iters 10

# Multiple-testing-corrected Deflated Sharpe (recommend N≥50 trials first)
OPTIMIZE_METRIC=dsr .venv/Scripts/python.exe loop.py --iters 10
```

The loop stops automatically after the requested iterations or three
consecutive regressions. Use the dashboard for a richer view.

### Pick what the loop optimizes on (`SYMBOLS`)

`SYMBOLS` is one knob: a comma-separated list of parquet stems under `data/`.
Mode is chosen by count — N=1 is single, N≥2 is basket (overfit-penalized).
Empty/unset defaults to `crypto/BTC_USDT_4h`.

```bash
# Single asset — strict 20-trade minimum, raw Sharpe
SYMBOLS=stocks/TSLA_1d .venv/Scripts/python.exe loop.py --iters 20

# Basket of stocks — penalizes strategies that win on one name and lose on others
# Score: mean(sharpe) - BASKET_PENALTY * std(sharpe)   (BASKET_PENALTY default 0.5)
for sym in TSLA NVDA AAPL MSFT GOOG; do
  .venv/Scripts/python.exe data_fetch.py --asset stocks --symbol $sym --timeframe 1d --start 2015-01-01
done
SYMBOLS="stocks/TSLA_1d,stocks/NVDA_1d,stocks/AAPL_1d,stocks/MSFT_1d,stocks/GOOG_1d" \
  .venv/Scripts/python.exe loop.py --iters 20
```

## Three time windows

| Window  | Range                   | Used by |
|---|---|---|
| `train` | 2019-01-01 → 2022-12-31 | Strategy reasoning only — never the metric |
| `val`   | 2023-01-01 → 2024-12-31 | What the loop optimizes against (default) |
| `lockbox` | 2025-01-01 → present  | **Held back.** Loop never sees it. Open manually before promoting a strategy. |

Run any window manually:

```bash
.venv/Scripts/python.exe backtest.py --window val
.venv/Scripts/python.exe backtest.py --window lockbox    # promotion check
```

The metrics block:

```
val_sharpe:       1.273990   ← backtesting.py reported (loop default)
sortino:          3.563552
sharpe_ann_4h:    1.728618   ← manually annualized (sqrt(periods/year))
calmar:           7.202247   ← total_return_pct / max_drawdown
psr:              0.993145   ← Probabilistic Sharpe Ratio (vs SR*=0)
dsr:              0.652134   ← Deflated Sharpe Ratio (vs SR*=trial-noise)
skew:             0.586
kurtosis:         18.400
max_drawdown:     18.05
win_rate:         0.467
total_trades:     30
total_return_pct: 130.03
```

## Daily signal scanner

`scan.py` reads the watchlist (`SCAN_WATCHLIST` env var), runs the current
`strategy.py` against recent daily bars for each symbol, and emits BUY/SELL
alerts when a signal fired on the most recent bar. Posts to a Discord/Slack
webhook if `SCAN_WEBHOOK_URL` is set, otherwise prints to stdout + appends
to `results/scan.log`.

```bash
.venv/Scripts/python.exe scan.py                    # use SCAN_WATCHLIST
.venv/Scripts/python.exe scan.py --symbols TSLA,NVDA --dry
```

This is the deployment surface for the "ping me when there's a signal,
I execute manually" workflow.

## GitHub Actions (scheduled runs)

Three workflows ship in `.github/workflows/`:

| Workflow | Schedule | What it does |
|---|---|---|
| `loop.yml` | Every 6h, matrix(stocks, crypto) | Runs the autoresearch loop for both campaigns in parallel; commits keeps + per-campaign tsv back to `main` |
| `scan.yml` | Stocks weekdays 13:30 UTC; crypto every 4h | Signal scanner posts to webhook |
| `paper.yml` | Stocks weekdays 13:35 UTC; crypto every 4h | Alpaca paper executor — runs `live_trade.py` against the latest `strategy.py` |

All three have `workflow_dispatch` triggers, so you can also fire them
on-demand from the GitHub Actions tab.

### Required repo secrets / variables

Settings → Secrets and variables → Actions:

| Type | Name | Used by | Purpose |
|---|---|---|---|
| Secret | `OPENROUTER_API_KEY` | `loop.yml` | OpenRouter API key (one key, any model — https://openrouter.ai/keys) |
| Secret | `SCAN_WEBHOOK_URL` | `scan.yml` | Discord/Slack webhook for alerts |
| Secret | `ALPACA_API_KEY` | `paper.yml` | Alpaca paper API key |
| Secret | `ALPACA_API_SECRET` | `paper.yml` | Alpaca paper API secret |
| Variable | `OPENROUTER_MODEL` | `loop.yml` | (optional) override model; any slug from https://openrouter.ai/models. Default `anthropic/claude-haiku-4-5` |
| Variable | `SCAN_WATCHLIST` | `scan.yml` | stocks, e.g. `SPY,QQQ,TSLA,NVDA` |
| Variable | `SCAN_WATCHLIST_CRYPTO` | `scan.yml` | crypto via yfinance, e.g. `BTC-USD,ETH-USD` |
| Variable | `PAPER_WATCHLIST` | `paper.yml` | stocks, Alpaca tickers |
| Variable | `PAPER_CRYPTO_WATCHLIST` | `paper.yml` | crypto, e.g. `BTC/USD,ETH/USD` |
| Variable | `PAPER_PER_SYMBOL_CASH` | `paper.yml` | per-symbol cash budget (default `10000`) |

## Paper trading (Alpaca, $10k per symbol)

The executor in `live_trade.py` reuses the same `backtesting.py` harness as
`scan.py`, so it automatically picks up whatever the autoresearch loop just
optimized. No hand-coded mirror of the strategy — there is one strategy.

```bash
# 1. Get free paper keys at https://app.alpaca.markets/paper/dashboard/overview
# 2. Add to .env:
#       ALPACA_API_KEY=...
#       ALPACA_API_SECRET=...
#       ALPACA_PAPER=True
#       PAPER_PER_SYMBOL_CASH=10000
#       PAPER_WATCHLIST=SPY,QQQ,TSLA,NVDA
#       PAPER_CRYPTO_WATCHLIST=BTC/USD,ETH/USD
# 3. Install the SDK
.venv/Scripts/python.exe -m pip install alpaca-py

# Dry run — fetch bars, run strategy, print signals, do NOT submit orders
.venv/Scripts/python.exe live_trade.py --dry --symbols SPY --asset stock

# Real paper run — submit market orders to the Alpaca paper endpoint
.venv/Scripts/python.exe live_trade.py --asset stock
.venv/Scripts/python.exe live_trade.py --asset crypto

# Live mode (real money) — must pass --live AND set ALPACA_PAPER=False
.venv/Scripts/python.exe live_trade.py --live --asset stock
```

The executor is idempotent — re-running it within the same bar checks for
existing positions and open orders before submitting, so the GitHub Actions
schedule won't double-submit. Every run appends a JSONL record to
`results/paper.log`.

## What the metrics actually mean

- **`val_sharpe`** — backtesting.py's reported Sharpe (auto-annualized from bar
  frequency). Default optimization target. Best understood as a "risk-adjusted
  return" score; higher is better; >1.0 is good for retail.
- **`calmar`** — `total_return / max_drawdown`. Closer to "dollars per dollar
  of pain." Picks strategies that compound without big drawdowns.
- **`psr`** — Probabilistic Sharpe Ratio (Bailey & López de Prado). The
  probability that the true Sharpe exceeds 0 given observed sample, accounting
  for skew/kurtosis. >0.95 = strong evidence the strategy isn't noise.
- **`dsr`** — Deflated Sharpe Ratio. Same idea but adjusted for the *number of
  experiments you've run* — corrects for the fact that running 100 trials and
  picking the best is a tournament, not alpha discovery. Don't trust until
  N ≥ 50 non-crash trials.
- **`sortino`** — like Sharpe but only penalizes downside volatility. Good
  sanity check that you're not just being penalized for upside spikes.

## Honest expectations

This is a hard game. The supporting research (see commit history for
`/research` and `/scout` outputs) finds that ~1–3% of retail algo traders are
consistently profitable. Realistic out-of-sample Sharpe ceiling is ~1.0;
expect 30–50% degradation from in-sample to out-of-sample.

Use the bot as a hobby with a small fraction of capital. Use index funds for
your retirement. The signal scanner's "alert me, I execute manually"
architecture is the most-validated retail pattern — that's why it's here.
