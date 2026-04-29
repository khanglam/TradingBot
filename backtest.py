"""Fixed evaluation harness — DO NOT MODIFY once stable.

The autoresearch loop calls this, parses the printed summary block, and uses
the configured OPTIMIZE_METRIC to decide keep-vs-discard. Changing this file
mid-experiment makes runs incomparable. Treat as immutable.

Usage:
    python backtest.py                          # default: configured asset, val window
    python backtest.py --data data/crypto/BTC_USDT_4h.parquet
    python backtest.py --window train           # eval on training window
    python backtest.py --window lockbox         # held-out 2025+ window (manual only)

Environment variables (read at runtime, override file defaults):
    BACKTEST_DATA       path to parquet (overrides DEFAULT_DATA / --data)
    BASKET_SYMBOLS      comma-separated list (e.g. "TSLA,NVDA,AAPL"). When
                        set, scores the strategy on each symbol independently
                        and aggregates into a single summary block.
                        Resolves to data/stocks/{SYMBOL}_{tf}.parquet
                        (timeframe from BASKET_TIMEFRAME, default "1d").
    BASKET_TIMEFRAME    timeframe suffix for basket lookup (default "1d")
    BASKET_ASSET_CLASS  "stocks" (default) or "crypto" — directory under data/
    BASKET_PENALTY      stdev penalty in basket scoring (default 0.5).
                        score = mean(sharpe) - penalty * std(sharpe).
    DSR_BENCHMARK       per-bar Sharpe benchmark for DSR (default 0; loop sets
                        this from prior-trial variance to correct for multiple
                        testing — see loop.py:compute_dsr_benchmark).

Windows:
  train   : 2019-01-01 → 2022-12-31  (agent reasons about it; not the metric)
  val     : 2023-01-01 → 2024-12-31  (loop optimizes against this)
  lockbox : 2025-01-01 → end-of-data (NEVER touched by loop; only inspected
            manually before promoting a strategy to paper trading)

Output (printed to stdout, parsed by loop.py):
    ---
    val_sharpe:        1.234567
    sortino:           1.890123
    sharpe_ann_4h:     1.728618
    calmar:            1.910234
    psr:               0.876543
    dsr:               0.654321
    skew:              0.123
    kurtosis:          5.4
    max_drawdown:      12.34
    win_rate:          0.456
    total_trades:      87
    total_return_pct:  45.67
    ---

In basket mode (BASKET_SYMBOLS set):
  - val_sharpe = mean(sharpes) - BASKET_PENALTY * std(sharpes)  ← penalize
    inconsistency across the basket so single-symbol overfit doesn't win.
  - sortino, sharpe_ann_4h, calmar, psr, dsr, skew, kurtosis, win_rate,
    total_return_pct = simple mean across surviving symbols.
  - max_drawdown = max (worst case across the basket).
  - total_trades = sum across the basket.

Crashes or insufficient trades print val_sharpe: 0.000000.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
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

ROOT = Path(__file__).parent

# ───────────────────────── Fixed configuration ──────────────────────────
# Changing any of these makes past results incomparable. Treat as constants.

DEFAULT_DATA = "data/crypto/BTC_USDT_4h.parquet"
# backtesting.py floors share counts to integers, so we use a large nominal
# cash so fractional-equity sizing produces >= 1 BTC unit. Sharpe and
# percentage returns are scale-invariant — this changes nothing about
# strategy comparison, just lets the broker actually fill orders.
STARTING_CASH = 1_000_000
COMMISSION = 0.0006  # 6 bps, KuCoin taker default for crypto

TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"
LOCKBOX_START = "2025-01-01"  # held-out; loop never touches this

MIN_TRADES = 20  # below this, val_sharpe forced to 0 (single-symbol mode only;
                 # basket mode aggregates total_trades and uses MIN_BASKET_TRADES)
MIN_BASKET_TRADES = 5  # per-symbol minimum in basket mode (looser since each
                       # symbol contributes ~1/N of the total signal)

# Annualization factor for 4h bars: 6 bars/day * 365 days = 2190.
# Used as the legacy default; per-run factor is inferred from data frequency.
ANN_FACTOR_4H = math.sqrt(365 * 6)


# ─────────────────────────── helpers ────────────────────────────────────

def _ann_factor_for(df: pd.DataFrame) -> float:
    """Annualization factor inferred from median bar spacing.
    Falls back to the 4h factor if the index is too short or unparseable."""
    try:
        if len(df) < 2:
            return ANN_FACTOR_4H
        delta = df.index.to_series().diff().median()
        secs = delta.total_seconds() if delta is not pd.NaT else 0
        if not secs or secs <= 0:
            return ANN_FACTOR_4H
        bars_per_year = (365.0 * 24 * 3600) / secs
        return math.sqrt(bars_per_year)
    except Exception:
        return ANN_FACTOR_4H


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


_ZERO_EXTRA = {"sharpe_ann_4h": 0.0, "calmar": 0.0, "psr": 0.0, "dsr": 0.0, "skew": 0.0, "kurtosis": 0.0}

ZERO_METRICS = {
    "val_sharpe": 0.0, "sortino": 0.0, "sharpe_ann_4h": 0.0, "calmar": 0.0,
    "psr": 0.0, "dsr": 0.0, "skew": 0.0, "kurtosis": 0.0,
    "max_drawdown": 0.0, "win_rate": 0.0, "total_trades": 0, "total_return_pct": 0.0,
}


def _extra_metrics(stats_obj, total_return: float, max_dd: float, ann_factor: float) -> dict:
    """Compute manually-annualized Sharpe, Calmar, PSR, DSR, skew, kurtosis
    from the equity curve. Falls back to zeros on any computation error.

    PSR uses sr_benchmark=0 (probability that true SR exceeds zero).
    DSR uses sr_benchmark=DSR_BENCHMARK env var (per-bar units, set by the
    loop from the variance of prior-trial Sharpes — corrects for the
    selection bias inherent in running many experiments)."""
    try:
        eq = stats_obj["_equity_curve"]["Equity"]
        rets = eq.pct_change().dropna()
        n = len(rets)
        if n < 2:
            return dict(_ZERO_EXTRA)

        mu = float(rets.mean())
        sd = float(rets.std(ddof=1))
        if sd <= 0:
            return dict(_ZERO_EXTRA)

        sharpe_per_bar = mu / sd
        sharpe_ann = sharpe_per_bar * ann_factor
        skew = float(sp_stats.skew(rets, bias=False))
        excess_kurt = float(sp_stats.kurtosis(rets, fisher=True, bias=False))
        psr = _psr(sharpe_per_bar, n, skew, excess_kurt, sr_benchmark=0.0)

        try:
            dsr_benchmark = float(os.environ.get("DSR_BENCHMARK", "0") or 0.0)
        except ValueError:
            dsr_benchmark = 0.0
        dsr = _psr(sharpe_per_bar, n, skew, excess_kurt, sr_benchmark=dsr_benchmark)

        # Calmar: rank-monotone within a fixed window. max_dd is in pct,
        # already absolute-valued upstream. Guard div-by-zero.
        calmar = (total_return / max_dd) if max_dd > 1e-9 else 0.0
        return {
            "sharpe_ann_4h": sharpe_ann,
            "calmar": calmar,
            "psr": psr,
            "dsr": dsr,
            "skew": skew,
            "kurtosis": excess_kurt,
        }
    except Exception as e:
        print(f"# extra-metrics error: {e}", file=sys.stderr)
        return dict(_ZERO_EXTRA)


# ─────────────────────────── single-symbol run ──────────────────────────

def _run_single(data_path: str | Path, window: str, min_trades: int) -> dict | None:
    """Run a single backtest. Returns metrics dict (or None on crash).
    Does not print; printing is handled at the caller level."""
    try:
        from backtesting import Backtest
        from strategy import Strategy as UserStrategy
    except Exception as e:
        print(f"# import error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"# data load: {e}", file=sys.stderr)
        return None

    df = _slice(df, window)
    if len(df) < 100:
        print(f"# only {len(df)} candles in {window} window for {data_path}", file=sys.stderr)
        return None

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
        print(f"# backtest run error on {data_path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

    n_trades = int(stats.get("# Trades", 0) or 0)
    # MIN_TRADES gate exists to make val-window Sharpe statistically meaningful
    # for the loop's keep/discard decision. For lockbox (manual inspection),
    # show whatever numbers we have — the user will judge sample-size adequacy.
    if window != "lockbox" and n_trades < min_trades:
        print(f"# only {n_trades} trades for {data_path} (min {min_trades})", file=sys.stderr)
        return None

    sharpe = float(stats.get("Sharpe Ratio", 0.0) or 0.0)
    sortino = float(stats.get("Sortino Ratio", 0.0) or 0.0)
    max_dd = abs(float(stats.get("Max. Drawdown [%]", 0.0) or 0.0))
    win_rate = float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0
    total_return = float(stats.get("Return [%]", 0.0) or 0.0)

    ann_factor = _ann_factor_for(df)
    extra = _extra_metrics(stats, total_return, max_dd, ann_factor)

    return {
        "val_sharpe": sharpe,
        "sortino": sortino,
        "sharpe_ann_4h": extra["sharpe_ann_4h"],
        "calmar": extra["calmar"],
        "psr": extra["psr"],
        "dsr": extra["dsr"],
        "skew": extra["skew"],
        "kurtosis": extra["kurtosis"],
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": n_trades,
        "total_return_pct": total_return,
    }


# ─────────────────────────── basket aggregation ─────────────────────────

def _basket_paths(symbols: list[str], asset_class: str, timeframe: str) -> list[tuple[str, Path]]:
    """Resolve symbol list to (symbol, parquet path) tuples."""
    sub = "stocks" if asset_class == "stocks" else "crypto"
    out: list[tuple[str, Path]] = []
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        # Crypto pairs may contain "/" — replace with underscore for filenames
        fname = sym.replace("/", "_")
        out.append((sym, ROOT / "data" / sub / f"{fname}_{timeframe}.parquet"))
    return out


def _aggregate_basket(per_symbol: list[tuple[str, dict]], penalty: float) -> dict:
    """Combine per-symbol metric dicts into a single basket result.

    Scoring: val_sharpe = mean(per-symbol sharpes) - penalty * std(per-symbol sharpes).
    The stdev term penalizes strategies that win on one symbol and lose on others
    (i.e., overfit to a single name) — this is the whole point of basket mode.
    """
    if not per_symbol:
        return dict(ZERO_METRICS)

    sharpes = [m["val_sharpe"] for _, m in per_symbol]
    sharpe_mean = statistics.mean(sharpes)
    sharpe_std = statistics.pstdev(sharpes) if len(sharpes) > 1 else 0.0

    def avg(key: str) -> float:
        return float(statistics.mean(m[key] for _, m in per_symbol))

    return {
        "val_sharpe": float(sharpe_mean - penalty * sharpe_std),
        "sortino": avg("sortino"),
        "sharpe_ann_4h": avg("sharpe_ann_4h"),
        "calmar": avg("calmar"),
        "psr": avg("psr"),
        "dsr": avg("dsr"),
        "skew": avg("skew"),
        "kurtosis": avg("kurtosis"),
        "max_drawdown": float(max(m["max_drawdown"] for _, m in per_symbol)),
        "win_rate": avg("win_rate"),
        "total_trades": int(sum(m["total_trades"] for _, m in per_symbol)),
        "total_return_pct": avg("total_return_pct"),
    }


def _run_basket(symbols: list[str], asset_class: str, timeframe: str, window: str, penalty: float) -> dict:
    """Run the strategy on each symbol in the basket and aggregate.

    Symbols whose data is missing or whose backtests crash are skipped with a
    warning; if at least one survives, the basket result is the aggregate of
    those that did. If none survive, returns zero metrics."""
    paths = _basket_paths(symbols, asset_class, timeframe)
    print(f"# basket mode: {len(paths)} symbols, asset={asset_class}, tf={timeframe}", file=sys.stderr)
    per_symbol: list[tuple[str, dict]] = []
    for sym, path in paths:
        if not path.exists():
            print(f"# basket: missing data for {sym} ({path}), skipping", file=sys.stderr)
            continue
        m = _run_single(path, window, min_trades=MIN_BASKET_TRADES)
        if m is None:
            print(f"# basket: {sym} crashed/insufficient, skipping", file=sys.stderr)
            continue
        print(
            f"# basket {sym}: sharpe={m['val_sharpe']:.4f} dd={m['max_drawdown']:.2f}% "
            f"trades={m['total_trades']} ret={m['total_return_pct']:.2f}%",
            file=sys.stderr,
        )
        per_symbol.append((sym, m))

    if not per_symbol:
        return dict(ZERO_METRICS)
    return _aggregate_basket(per_symbol, penalty)


# ─────────────────────────── public run() ───────────────────────────────

def run(data_path: str | Path = DEFAULT_DATA, window: str = "val") -> dict:
    """Single entrypoint. Routes to basket mode when BASKET_SYMBOLS is set;
    otherwise runs a standard single-symbol backtest."""
    basket_env = (os.environ.get("BASKET_SYMBOLS") or "").strip()

    if basket_env:
        symbols = [s for s in basket_env.split(",") if s.strip()]
        timeframe = (os.environ.get("BASKET_TIMEFRAME") or "1d").strip() or "1d"
        asset_class = (os.environ.get("BASKET_ASSET_CLASS") or "stocks").strip() or "stocks"
        try:
            penalty = float(os.environ.get("BASKET_PENALTY", "0.5") or 0.5)
        except ValueError:
            penalty = 0.5
        metrics = _run_basket(symbols, asset_class, timeframe, window, penalty)
    else:
        metrics = _run_single(data_path, window, min_trades=MIN_TRADES)
        if metrics is None:
            metrics = dict(ZERO_METRICS)

    _print_summary(metrics)
    return metrics


def _print_summary(m: dict) -> None:
    """Print the summary block in the canonical order the loop parses."""
    print("---")
    print(f"{'val_sharpe:':<18}{m['val_sharpe']:.6f}")
    print(f"{'sortino:':<18}{m['sortino']:.6f}")
    print(f"{'sharpe_ann_4h:':<18}{m['sharpe_ann_4h']:.6f}")
    print(f"{'calmar:':<18}{m['calmar']:.6f}")
    print(f"{'psr:':<18}{m['psr']:.6f}")
    print(f"{'dsr:':<18}{m['dsr']:.6f}")
    print(f"{'skew:':<18}{m['skew']:.3f}")
    print(f"{'kurtosis:':<18}{m['kurtosis']:.3f}")
    print(f"{'max_drawdown:':<18}{m['max_drawdown']:.2f}")
    print(f"{'win_rate:':<18}{m['win_rate']:.3f}")
    print(f"{'total_trades:':<18}{m['total_trades']}")
    print(f"{'total_return_pct:':<18}{m['total_return_pct']:.2f}")
    print("---")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("BACKTEST_DATA") or DEFAULT_DATA)
    p.add_argument("--window", choices=["train", "val", "lockbox"], default="val")
    args = p.parse_args()
    run(args.data, args.window)
