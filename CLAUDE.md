# TradingBot — Autonomous Strategy Optimization

## What This Project Is

This is a **karpathy-autoresearch-style autonomous trading strategy optimizer**. The idea: an AI agent modifies a trading strategy file, runs a backtest, measures a performance metric, keeps improvements, discards regressions, and loops forever — exactly like autoresearch does for LLM training.

Reference: `karpathy-auto-research/` in this repo is the original. Read its `program.md` to understand the loop pattern we're replicating.

The framework we build on is **Jesse** (`jesse/`), chosen because:
- Its `backtest()` is a pure Python function — callable without CLI, no subprocess spawning
- Strategy API is minimal: extend `Strategy`, implement `should_long()` / `should_exit_long()` / `go_long()` / `go_short()`
- Built-in Optuna hyperparameter optimization and Monte Carlo robustness checks
- No look-ahead bias by design
- Rust performance layer for fast backtesting iteration

---

## The Autoresearch Loop (Design Target)

```
program.md          ← agent instructions (written by human, not AI)
      ↓
LOOP FOREVER:
  1. Read current strategy.py and results.tsv
  2. Modify strategy.py with an experimental idea
  3. git commit
  4. Run backtest → run.log
  5. Read val_sharpe from run.log
  6. If improved: keep commit (advance branch)
     If worse: git reset to last good commit
  7. Log to results.tsv
  8. Repeat
```

This maps directly to autoresearch:

| karpathy-autoresearch | This project |
|---|---|
| `prepare.py` (fixed) | `backtest.py` — immutable harness |
| `train.py` (mutable) | `strategy.py` — only file agent edits |
| `val_bpb` (lower = better) | `val_sharpe` (higher = better) |
| Fixed 5-min time budget | Fixed backtest date window |
| `results.tsv` | `results.tsv` (same format) |
| `program.md` | `program.md` (to be written) |

---

## Repository Layout

```
TradingBot/
├── CLAUDE.md                    ← this file
├── jesse/                       ← Jesse framework (cloned, do not modify internals)
│   ├── jesse/
│   │   ├── modes/
│   │   │   └── backtest_mode.py ← backtest engine entry point
│   │   ├── strategies/
│   │   │   └── Strategy.py      ← base class all strategies inherit
│   │   └── indicators/          ← technical indicators (ta, numpy-based)
│   └── tests/
├── karpathy-auto-research/      ← reference implementation, read-only
│
│   ── FILES WE BUILD ──
│
├── backtest.py                  ← FIXED harness (like prepare.py) — do not modify once stable
├── strategy.py                  ← MUTABLE file the agent edits (like train.py)
├── program.md                   ← agent instructions for the loop
├── results.tsv                  ← experiment log (untracked by git)
└── run.log                      ← latest backtest output (untracked by git)
```

---

## The Two Sacred Files

### `backtest.py` — Fixed Harness (DO NOT MODIFY once stable)
This is the evaluation oracle. It:
- Loads historical candle data for a fixed asset and date range
- Instantiates and runs the strategy from `strategy.py`
- Computes the evaluation metric
- Prints a summary block the agent can parse

The agent **never touches this file**. Changing it would make experiments incomparable — same as modifying `prepare.py` in autoresearch.

### `strategy.py` — Mutable Strategy (Agent edits this)
This is the only file the agent modifies. It extends Jesse's `Strategy` base class and implements signal logic. Everything is fair game: entry/exit conditions, indicators used, position sizing, stop-loss logic, hyperparameters.

---

## Evaluation Metric

Primary metric: **`val_sharpe`** (annualized Sharpe ratio on the validation window, higher is better).

Secondary constraints (soft limits, not primary objectives):
- `max_drawdown` < 30% (hard stop — a strategy that blows up is not acceptable regardless of Sharpe)
- `win_rate` > 30% (sanity check — pure luck strategies filtered out)
- `total_trades` > 20 (minimum sample size for the metric to be meaningful)

The backtest window is split:
- **Train window**: used implicitly via the optimization loop
- **Validation window**: fixed, out-of-sample — `val_sharpe` is always reported on this

---

## Backtest Configuration (to be finalized)

| Parameter | Value |
|-----------|-------|
| Asset | BTC-USDT (start with single asset) |
| Exchange | Binance Futures (simulated) |
| Timeframe | 4h |
| Train window | 2019-01-01 → 2022-12-31 |
| Validation window | 2023-01-01 → 2024-12-31 |
| Starting capital | $10,000 |
| Leverage | 1x (no leverage initially) |

---

## Output Format

`backtest.py` must print a summary block that the agent can parse with grep:

```
---
val_sharpe:        1.234567
max_drawdown:      12.34
win_rate:          0.456
total_trades:      87
total_return_pct:  45.67
```

The agent reads metrics with:
```bash
grep "^val_sharpe:" run.log
grep "^max_drawdown:" run.log
```

If the run crashes or produces insufficient trades, val_sharpe is reported as 0.0.

---

## Results Log (`results.tsv`)

Tab-separated, NOT comma-separated. Never committed to git.

```
commit	val_sharpe	max_drawdown	status	description
a1b2c3d	1.234567	12.3	keep	baseline — simple EMA crossover
b2c3d4e	1.456789	10.1	keep	added RSI filter
c3d4e5f	0.987654	18.2	discard	tried Bollinger Band exits
d4e5f6g	0.000000	0.0	crash	look-ahead bias error, reverted
```

Columns:
1. `commit` — 7-char git hash
2. `val_sharpe` — validation Sharpe (0.000000 for crashes)
3. `max_drawdown` — peak drawdown % (0.0 for crashes)
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short plain-English summary of what was tried

---

## Jesse Strategy API (Quick Reference)

The agent needs to know how to write valid Jesse strategies:

```python
from jesse.strategies import Strategy
import jesse.indicators as ta

class MyStrategy(Strategy):

    # Hyperparameters (optimizable)
    @property
    def hp(self):
        return [
            {'name': 'ema_period', 'type': int, 'min': 10, 'max': 200, 'default': 50},
        ]

    # Required: define entry conditions
    def should_long(self) -> bool:
        return ta.ema(self.candles, self.hp['ema_period']) > ...

    def should_short(self) -> bool:
        return False  # disable shorts initially

    def should_cancel_entry(self) -> bool:
        return False

    # Required: define entry orders
    def go_long(self):
        self.buy = self.available_margin, self.price  # market order

    def go_short(self):
        pass

    # Required: define exit conditions
    def update_position(self):
        if ...:  # exit condition
            self.liquidate()
```

Key properties available inside strategy:
- `self.candles` — OHLCV numpy array (shape: [n, 6])
- `self.price` / `self.close` — current close price
- `self.open`, `self.high`, `self.low`, `self.volume`
- `self.position` — current open position
- `self.available_margin` — available capital
- `self.hp` — hyperparameter dict

Key indicators in `jesse.indicators` (aliased as `ta`):
- `ta.ema(candles, period)`, `ta.sma(candles, period)`
- `ta.rsi(candles, period)`, `ta.macd(candles)`
- `ta.bollinger_bands(candles, period)`, `ta.atr(candles, period)`
- `ta.stoch(candles)`, `ta.adx(candles, period)`

---

## Simplicity Criterion

Identical to autoresearch: all else equal, simpler is better.

- A 0.01 Sharpe improvement that adds 50 lines of complex logic? Not worth it.
- A 0.01 Sharpe improvement from deleting code? Keep it.
- Removing an indicator and getting equal performance? That's a win.

The agent should prefer strategies that are readable, robust, and minimal.

---

## What the Agent Can and Cannot Do

**CAN modify:**
- `strategy.py` — anything: entry/exit logic, indicators, position sizing, stop-loss, hyperparameters

**CANNOT modify:**
- `backtest.py` — the fixed evaluation harness
- Jesse internals (`jesse/`) — treat as a read-only dependency
- The validation date window — changing it makes results incomparable
- `results.tsv` — append only, never edit past rows

---

## Skills Available in This Project

Three global skills are installed for research and decision-making:

- `/research <topic>` — fan-out parallel research (5 agents, different angles)
- `/debate <claim>` — stochastic multi-agent consensus (3 agents, adversarial roles)
- `/scout <topic>` — full research → debate pipeline in one command

Use `/scout` before adopting any new framework, library, or major architectural decision.

---

## Current Status

- [x] Jesse cloned to `jesse/`
- [x] karpathy-auto-research in `karpathy-auto-research/` for reference
- [ ] `backtest.py` — not yet written
- [ ] `strategy.py` — not yet written (baseline)
- [ ] `program.md` — not yet written
- [ ] `results.tsv` — not yet initialized
- [ ] Jesse installed and data downloaded

## Next Steps

1. Install Jesse and verify backtest runs
2. Write `backtest.py` (fixed harness) — callable from Python, prints the summary block
3. Write `strategy.py` (baseline — simple EMA crossover to establish baseline metric)
4. Run baseline backtest, record in `results.tsv`
5. Write `program.md` (agent instructions for the loop)
6. Launch the autonomous loop
