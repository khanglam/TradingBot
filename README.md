# TradingBot

A self-improving trading strategy. An LLM proposes mutations to `strategy.py`,
the harness runs a backtest, and only improvements are kept. Loops forever.

## Files

| File | Role |
|---|---|
| `strategy.py` | The trading strategy. **The agent edits this and only this.** |
| `backtest.py` | Fixed evaluation harness. Runs the strategy on the validation window and prints metrics. **Don't modify after baseline.** |
| `loop.py` | The autoresearch orchestrator. Asks Claude for a mutation → backtests → keeps wins (git commit) or reverts (git reset). |
| `data_fetch.py` | Downloads OHLCV data (BTC/USDT from KuCoin, stocks from yfinance) → Parquet files in `data/`. |
| `live_trade.py` | Paper / live execution via LumiBot + Alpaca. Same `strategy.py`, no rewrite needed. |
| `app.py` | The dashboard server (FastAPI + a managed loop subprocess + SSE for live output). |
| `index.html` | The dashboard UI — single self-contained file (Tailwind + Alpine + Chart.js, all from CDNs). |
| `program.md` | Constraints and required output format the LLM mutator must follow. |
| `results.tsv` | Append-only experiment log (gitignored). |
| `CLAUDE.md` | Full project design notes. |

## Run the dashboard

```bash
.venv/Scripts/python.exe app.py
```

Then open http://127.0.0.1:8000. Press `Ctrl+C` to stop.

The dashboard shows: best Sharpe so far, equity curve vs Buy & Hold, drawdown,
Sharpe progression, the experiments table, the current `strategy.py` source,
git history, and a launch button for the loop with a live console.

## First-time setup

```bash
# 1. Install Python dependencies into the venv
.venv/Scripts/python.exe -m pip install \
    backtesting ccxt yfinance anthropic pyarrow \
    fastapi "uvicorn[standard]" sse-starlette

# 2. Download historical data (~10 seconds, ~14k rows for 6 years of 4h candles)
.venv/Scripts/python.exe data_fetch.py --asset crypto --symbol BTC/USDT --timeframe 4h --start 2019-01-01 --end 2024-12-31

# 3. Put your Anthropic API key in a .env file at project root
echo ANTHROPIC_API_KEY=sk-ant-... > .env

# 4. Make sure the git tree is clean (loop.py refuses to start if dirty)
git add -A && git commit -m "scaffold"
```

## Running the loop

Two ways:

**From the dashboard** — click **▶ Launch loop** in the top-right, set iterations,
press **▶ Start**. Watch the console at the bottom. KPIs and charts update live as
keeps land.

**From the terminal**:

```bash
.venv/Scripts/python.exe loop.py --iters 10
```

The loop stops automatically after the requested iterations, or sooner if there
are 3 consecutive regressions.

## Single backtest

```bash
.venv/Scripts/python.exe backtest.py --window val
```

Prints a metrics block:

```
val_sharpe:       0.961494
sortino:          2.500493
max_drawdown:     28.72
win_rate:         0.297
total_trades:     37
total_return_pct: 125.19
```

The dashboard also has a **▶ run backtest** button next to the strategy code.

## Paper trading

```bash
.venv/Scripts/python.exe -m pip install lumibot   # one-time, heavy
# Add to .env:
#   ALPACA_API_KEY=...
#   ALPACA_API_SECRET=...
#   ALPACA_PAPER=True
.venv/Scripts/python.exe live_trade.py --symbol SPY --asset stock
```

Flip `ALPACA_PAPER=False` (or pass `--live`) to use real money. **Don't.**
