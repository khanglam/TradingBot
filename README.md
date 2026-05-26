# TradingBot

A self-improving trading strategy. An LLM proposes mutations to per-campaign
`strategies/stocks.py` and `strategies/crypto.py`, the harness runs a backtest,
and only improvements are kept on **`dev`**. Each morning **`main`** is updated by
merging **`dev`** (see `sync_branches.yml`) for Alpaca paper trading.

## Branches

| Branch | Role |
|--------|------|
| `dev` | Research — loop mutates strategies here |
| `main` | Deploy branch — daily merge from `dev`; paper reads this |

See [MIGRATION.md](MIGRATION.md) for one-time setup from the old `autoresearch/*` branches.

## Layout

```
TradingBot/
├── strategies/
│   ├── stocks.py           ← candidate on dev; frozen copy on main after promotion
│   └── crypto.py
├── backtest.py             ← evaluation harness
├── loop.py                 ← orchestrator (runs on dev only)
├── live_trade.py           ← Alpaca paper executor (reads main)
├── data_fetch.py           ← OHLCV downloader
├── app.py                  ← FastAPI dashboard (run from dev checkout)
├── program.md              ← LLM system prompt
├── web/index.html          ← dashboard UI
├── results/                ← experiment logs (*.tsv on dev)
├── data/                   ← cached parquet OHLCV (gitignored)
└── .github/workflows/
    ├── loop-stocks.yml     ← daily 03:00 PDT on dev (disable in Actions UI)
    ├── loop-crypto.yml     ← daily 03:00 PDT on dev (disable in Actions UI)
    ├── sync_branches.yml   ← daily 06:00 PDT: merge dev → main
    └── paper.yml           ← Alpaca paper on main
```

## First-time setup

Use `/init-local-dev` or manually:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip on macOS/Linux

git fetch origin dev
git checkout dev

cp .env.example .env   # add OPENROUTER_API_KEY
```

Fetch OHLCV (example):

```bash
.venv/Scripts/python.exe data_fetch.py --asset crypto --symbol BTC/USDT --timeframe 4h --start 2019-01-01
```

## Run the dashboard

From a **`dev`** checkout:

```bash
.venv/Scripts/python.exe app.py
# http://127.0.0.1:8000
```

## Running the autoresearch loop

From **`dev`**, clean git tree:

```bash
.venv/Scripts/python.exe loop.py --iters 10
OPTIMIZE_METRIC=calmar .venv/Scripts/python.exe loop.py --iters 10
```

Avoid running locally at **03:00 PDT** when GitHub Actions pushes to `dev`.

## Time windows

Configured in `configs.toml` (rolled forward May 2026). Loop scores **`val` only**.

| Window | Crypto | Stocks |
|--------|--------|--------|
| `train` | 2019 → 2022 | 2018 → 2021 |
| `val` | 2023 → 2025 | 2022 → 2025 |
| `lockbox` | 2026 → present | 2026 → present |

```bash
.venv/Scripts/python.exe backtest.py --window val
.venv/Scripts/python.exe backtest.py --window lockbox
```

## GitHub Actions

| Workflow | Schedule | Branch |
|----------|----------|--------|
| `loop-stocks.yml` | 10:00 UTC daily (03:00 PDT) | `dev` |
| `loop-crypto.yml` | 10:00 UTC daily (03:00 PDT) | `dev` |
| `sync_branches.yml` | 13:00 UTC daily (06:00 PDT) | Merge `dev` → `main` |
| `paper.yml` | Stocks weekdays 13:35 UTC; crypto every 4h | `main` |

Secrets: `OPENROUTER_API_KEY` (loop), `ALPACA_API_KEY` / `ALPACA_API_SECRET` (paper).

## Paper trading (Alpaca)

Runs against **frozen** strategies on `main` (checkout `main` locally or let CI run `paper.yml`):

```bash
git checkout main
.venv/Scripts/python.exe live_trade.py --dry --symbols SPY --asset stock
.venv/Scripts/python.exe live_trade.py --asset stock
```

See `.env.example` for `PAPER_*` variables.
