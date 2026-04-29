"""Fixed evaluation harness — DO NOT MODIFY once stable.

The autoresearch loop calls this, parses the printed summary block, and uses
val_sharpe to decide keep-vs-discard. Changing this file mid-experiment makes
runs incomparable. Treat as immutable.

Usage:
    python backtest.py                          # default: BTC/USDT 4h, validation window
    python backtest.py --data data/crypto/BTC_USDT_4h.parquet
    python backtest.py --window train           # eval on training window instead

Output (printed to stdout, parsed by loop.py):
    ---
    val_sharpe:        1.234567
    sortino:           1.890123
    max_drawdown:      12.34
    win_rate:          0.456
    total_trades:      87
    total_return_pct:  45.67
    ---

Crashes or insufficient trades print val_sharpe: 0.000000.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ───────────────────────── Fixed configuration ──────────────────────────
# Changing any of these makes past results incomparable. Treat as constants.

DEFAULT_DATA = "data/crypto/BTC_USDT_4h.parquet"
# backtesting.py floors share counts to integers, so we use a large nominal
# cash so fractional-equity sizing produces >= 1 BTC unit. Sharpe and
# percentage returns are scale-invariant — this changes nothing about
# strategy comparison, just lets the broker actually fill orders.
STARTING_CASH = 1_000_000
COMMISSION = 0.0006  # 6 bps, KuCoin taker (was Binance, swapped due to US 451)

TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"

MIN_TRADES = 20  # below this, val_sharpe forced to 0


def _zero_summary(reason: str) -> None:
    print(f"# crash/insufficient: {reason}", file=sys.stderr)
    print("---")
    print(f"{'val_sharpe:':<18}0.000000")
    print(f"{'sortino:':<18}0.000000")
    print(f"{'max_drawdown:':<18}0.00")
    print(f"{'win_rate:':<18}0.000")
    print(f"{'total_trades:':<18}0")
    print(f"{'total_return_pct:':<18}0.00")
    print("---")


def _slice(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "train":
        return df.loc[TRAIN_START:TRAIN_END]
    if window == "val":
        return df.loc[VAL_START:VAL_END]
    raise ValueError(f"unknown window: {window}")


def run(data_path: str | Path = DEFAULT_DATA, window: str = "val") -> dict:
    """Run a single backtest. Returns metrics dict and prints the summary block."""
    try:
        from backtesting import Backtest
        from strategy import Strategy as UserStrategy
    except Exception as e:
        _zero_summary(f"import error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {}

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        _zero_summary(f"data load: {e}")
        return {}

    df = _slice(df, window)
    if len(df) < 100:
        _zero_summary(f"only {len(df)} candles in {window} window")
        return {}

    try:
        bt = Backtest(
            df,
            UserStrategy,
            cash=STARTING_CASH,
            commission=COMMISSION,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
    except Exception as e:
        _zero_summary(f"backtest run: {e}")
        traceback.print_exc(file=sys.stderr)
        return {}

    n_trades = int(stats.get("# Trades", 0) or 0)
    if n_trades < MIN_TRADES:
        _zero_summary(f"only {n_trades} trades (min {MIN_TRADES})")
        return {}

    sharpe = float(stats.get("Sharpe Ratio", 0.0) or 0.0)
    sortino = float(stats.get("Sortino Ratio", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0
    total_return = float(stats.get("Return [%]", 0.0) or 0.0)

    metrics = {
        "val_sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": abs(max_dd),
        "win_rate": win_rate,
        "total_trades": n_trades,
        "total_return_pct": total_return,
    }

    print("---")
    print(f"{'val_sharpe:':<18}{metrics['val_sharpe']:.6f}")
    print(f"{'sortino:':<18}{metrics['sortino']:.6f}")
    print(f"{'max_drawdown:':<18}{metrics['max_drawdown']:.2f}")
    print(f"{'win_rate:':<18}{metrics['win_rate']:.3f}")
    print(f"{'total_trades:':<18}{metrics['total_trades']}")
    print(f"{'total_return_pct:':<18}{metrics['total_return_pct']:.2f}")
    print("---")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--window", choices=["train", "val"], default="val")
    args = p.parse_args()
    run(args.data, args.window)
