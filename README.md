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
├── live_trade.py           ← Optional Alpaca paper/live executor (LumiBot-based)
├── data_fetch.py           ← OHLCV downloader (CCXT for crypto, yfinance for stocks)
├── app.py                  ← FastAPI dashboard server
├── program.md              ← System prompt: rules + output format for the LLM
├── web/index.html          ← Dashboard UI (single self-contained file)
├── results/                ← Runtime outputs
│   ├── results.tsv         ← Append-only experiment log (committed)
│   ├── run.log             ← Latest backtest stdout (gitignored)
│   └── scan.log            ← Append-only signal-scan log (gitignored)
├── data/                   ← Cached parquet OHLCV (gitignored)
├── archive/                ← Old code retained for reference
└── .github/workflows/
    ├── loop.yml            ← Scheduled autoresearch run (weekly Sunday 02:00 PST)
    └── scan.yml            ← Scheduled signal scan (weekdays 05:30 PST)
```

## First-time setup

```bash
# 1. Install Python dependencies
.venv/Scripts/python.exe -m pip install \
    backtesting ccxt yfinance anthropic pyarrow scipy python-dotenv \
    fastapi "uvicorn[standard]" sse-starlette

# 2. Download historical data (4h BTC for the default loop)
.venv/Scripts/python.exe data_fetch.py --asset crypto --symbol BTC/USDT --timeframe 4h --start 2019-01-01

# 3. Copy .env.example → .env and fill in ANTHROPIC_API_KEY
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

Two workflows ship in `.github/workflows/`:

| Workflow | Schedule | What it does |
|---|---|---|
| `loop.yml` | Sunday 02:00 PST | Runs the autoresearch loop, commits keeps + results.tsv back to main |
| `scan.yml` | Weekdays 05:30 PST | Runs the signal scanner, posts to webhook |

Both have `workflow_dispatch` triggers, so you can also fire them
on-demand from the GitHub Actions tab.

### Required repo secrets / variables

Settings → Secrets and variables → Actions:

| Type | Name | Used by | Purpose |
|---|---|---|---|
| Secret | `ANTHROPIC_API_KEY` | `loop.yml` | Claude API key |
| Secret | `SCAN_WEBHOOK_URL` | `scan.yml` | Discord/Slack webhook for alerts |
| Variable | `CLAUDE_MODEL` | `loop.yml` | (optional) override model; default `claude-haiku-4-5` |
| Variable | `SCAN_WATCHLIST` | `scan.yml` | comma-sep symbols, e.g. `SPY,QQQ,TSLA,NVDA` |
| Variable | `SCAN_LOOKBACK_DAYS` | `scan.yml` | (optional) default 250 |

## Paper trading

```bash
.venv/Scripts/python.exe -m pip install lumibot
# In .env:
#   ALPACA_API_KEY=...
#   ALPACA_API_SECRET=...
#   ALPACA_PAPER=True
.venv/Scripts/python.exe live_trade.py --symbol SPY --asset stock
```

Flip `ALPACA_PAPER=False` (or pass `--live`) to use real money. **Don't.**

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
