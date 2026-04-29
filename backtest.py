"""Fixed evaluation harness — DO NOT MODIFY once stable.

The autoresearch loop calls this, parses the printed summary block, and uses
the configured OPTIMIZE_METRIC to decide keep-vs-discard. Changing this file
mid-experiment makes runs incomparable. Treat as immutable.

Usage:
    python backtest.py                          # default: BTC/USDT 4h, validation window
    python backtest.py --data data/crypto/BTC_USDT_4h.parquet
    python backtest.py --window train           # eval on training window
    python backtest.py --window lockbox         # held-out 2025+ window (manual only)

Windows:
  train   : 2019-01-01 → 2022-12-31  (agent reasons about it; not the metric)
  val     : 2023-01-01 → 2024-12-31  (loop optimizes against this)
  lockbox : 2025-01-01 → end-of-data (NEVER touched by loop; only inspected
            manually before promoting a strategy to paper trading)

Output (printed to stdout, parsed by loop.py):
    ---
    val_sharpe:        1.234567   ← backtesting.py reported (loop default)
    sortino:           1.890123   ← backtesting.py reported
    sharpe_ann_4h:     1.728618   ← manually annualized at sqrt(365*6)
    calmar:            1.910234   ← total_return_pct / max_drawdown
    psr:               0.876543   ← Probabilistic Sharpe Ratio vs SR*=0
    skew:              0.123      ← bar-level return skew
    kurtosis:          5.4        ← bar-level excess kurtosis
    max_drawdown:      12.34
    win_rate:          0.456
    total_trades:      87
    total_return_pct:  45.67
    ---

NOTE on annualization: backtesting.py auto-detects bar frequency and applies
its own annualization to "Sharpe Ratio" / "Sortino Ratio". Empirically on
4h BTC data, its number sits between sqrt(252) and sqrt(2190) — consistent
across runs (so rankings are preserved) but not directly comparable to
academic literature. `sharpe_ann_4h` is the manually annualized version at
the textbook sqrt(365*6)=46.8 factor for absolute interpretability.

Crashes or insufficient trades print val_sharpe: 0.000000.
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

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
LOCKBOX_START = "2025-01-01"  # held-out; loop never touches this

MIN_TRADES = 20  # below this, val_sharpe forced to 0

# Annualization factor for 4h bars: 6 bars/day * 365 days = 2190
ANN_FACTOR_4H = math.sqrt(365 * 6)


def _zero_summary(reason: str) -> None:
    print(f"# crash/insufficient: {reason}", file=sys.stderr)
    print("---")
    print(f"{'val_sharpe:':<18}0.000000")
    print(f"{'sortino:':<18}0.000000")
    print(f"{'sharpe_ann_4h:':<18}0.000000")
    print(f"{'calmar:':<18}0.000000")
    print(f"{'psr:':<18}0.000000")
    print(f"{'skew:':<18}0.000")
    print(f"{'kurtosis:':<18}0.000")
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
    if window == "lockbox":
        return df.loc[LOCKBOX_START:]
    raise ValueError(f"unknown window: {window}")


def _psr(sharpe_per_bar: float, n: int, skew: float, excess_kurt: float, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio — Bailey & López de Prado 2012.

    Probability that the true Sharpe exceeds sr_benchmark, given observed
    sharpe, sample size, skew, and excess kurtosis. SR is at the bar
    frequency, NOT annualized — PSR is dimensionless via the sqrt(T-1) term.
    Returns 0 if sample is degenerate.
    """
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sharpe_per_bar + (excess_kurt / 4.0) * sharpe_per_bar ** 2
    if denom_sq <= 0:
        return 0.0
    z = (sharpe_per_bar - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(sp_stats.norm.cdf(z))


def _extra_metrics(stats_obj, total_return: float, max_dd: float) -> dict:
    """Compute manually-annualized Sharpe, Calmar, PSR, skew, kurtosis from
    the equity curve. Falls back to zeros on any computation error."""
    try:
        eq = stats_obj["_equity_curve"]["Equity"]
        rets = eq.pct_change().dropna()
        n = len(rets)
        if n < 2:
            return {"sharpe_ann_4h": 0.0, "calmar": 0.0, "psr": 0.0, "skew": 0.0, "kurtosis": 0.0}

        mu = float(rets.mean())
        sd = float(rets.std(ddof=1))
        if sd <= 0:
            return {"sharpe_ann_4h": 0.0, "calmar": 0.0, "psr": 0.0, "skew": 0.0, "kurtosis": 0.0}

        sharpe_per_bar = mu / sd
        sharpe_ann = sharpe_per_bar * ANN_FACTOR_4H
        skew = float(sp_stats.skew(rets, bias=False))
        excess_kurt = float(sp_stats.kurtosis(rets, fisher=True, bias=False))
        psr = _psr(sharpe_per_bar, n, skew, excess_kurt, sr_benchmark=0.0)
        # Calmar: rank-monotone within a fixed window. max_dd is in pct,
        # already absolute-valued upstream. Guard div-by-zero.
        calmar = (total_return / max_dd) if max_dd > 1e-9 else 0.0
        return {
            "sharpe_ann_4h": sharpe_ann,
            "calmar": calmar,
            "psr": psr,
            "skew": skew,
            "kurtosis": excess_kurt,
        }
    except Exception as e:
        print(f"# extra-metrics error: {e}", file=sys.stderr)
        return {"sharpe_ann_4h": 0.0, "calmar": 0.0, "psr": 0.0, "skew": 0.0, "kurtosis": 0.0}


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
    # MIN_TRADES gate exists to make val-window Sharpe statistically meaningful
    # for the loop's keep/discard decision. For lockbox (manual inspection),
    # show whatever numbers we have — the user will judge sample-size adequacy.
    if window != "lockbox" and n_trades < MIN_TRADES:
        _zero_summary(f"only {n_trades} trades (min {MIN_TRADES})")
        return {}

    sharpe = float(stats.get("Sharpe Ratio", 0.0) or 0.0)
    sortino = float(stats.get("Sortino Ratio", 0.0) or 0.0)
    max_dd = abs(float(stats.get("Max. Drawdown [%]", 0.0) or 0.0))
    win_rate = float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0
    total_return = float(stats.get("Return [%]", 0.0) or 0.0)

    extra = _extra_metrics(stats, total_return, max_dd)

    metrics = {
        "val_sharpe": sharpe,
        "sortino": sortino,
        "sharpe_ann_4h": extra["sharpe_ann_4h"],
        "calmar": extra["calmar"],
        "psr": extra["psr"],
        "skew": extra["skew"],
        "kurtosis": extra["kurtosis"],
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": n_trades,
        "total_return_pct": total_return,
    }

    print("---")
    print(f"{'val_sharpe:':<18}{metrics['val_sharpe']:.6f}")
    print(f"{'sortino:':<18}{metrics['sortino']:.6f}")
    print(f"{'sharpe_ann_4h:':<18}{metrics['sharpe_ann_4h']:.6f}")
    print(f"{'calmar:':<18}{metrics['calmar']:.6f}")
    print(f"{'psr:':<18}{metrics['psr']:.6f}")
    print(f"{'skew:':<18}{metrics['skew']:.3f}")
    print(f"{'kurtosis:':<18}{metrics['kurtosis']:.3f}")
    print(f"{'max_drawdown:':<18}{metrics['max_drawdown']:.2f}")
    print(f"{'win_rate:':<18}{metrics['win_rate']:.3f}")
    print(f"{'total_trades:':<18}{metrics['total_trades']}")
    print(f"{'total_return_pct:':<18}{metrics['total_return_pct']:.2f}")
    print("---")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--window", choices=["train", "val", "lockbox"], default="val")
    args = p.parse_args()
    run(args.data, args.window)
