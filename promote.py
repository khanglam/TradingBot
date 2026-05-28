"""Lockbox promotion gate — the only sanctioned consumer of the held-out window.

Runs the current HEAD strategy on the lockbox slice exactly once, evaluates
against fixed gates, and writes one audit row to results/promotions.tsv.
live_trade.py refuses to start unless the current HEAD has a PASS row here.

Why a separate script:
  1. Every call to lockbox is auditable — line count in promotions.tsv == peeks.
  2. The loop has no import path to lockbox metrics, so the holdout stays blind
     to iteration.
  3. If you find yourself re-running promote.py to "tune" lockbox, you've
     burned the holdout — start a new one with a later lockbox_start.

Usage:
    python promote.py                                  # promote current HEAD
    python promote.py --symbols crypto/BTC_USDT_4h     # override symbols
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import backtest

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PROMOTIONS = RESULTS_DIR / "promotions.tsv"
PROMOTIONS_COLS = [
    "timestamp", "commit", "symbols",
    "lockbox_sharpe", "lockbox_max_dd", "lockbox_trades", "lockbox_bars",
    "decision", "reasons",
]
PROMOTIONS_HEADER = "\t".join(PROMOTIONS_COLS) + "\n"

# Gates — read at import time. Reuses the same MIN_TRADES / MAX_DRAWDOWN_LIMIT
# the loop uses so promotion criteria don't drift from optimization criteria.
MIN_TRADES = int(backtest._cfg("MIN_TRADES", "20"))
MAX_DD_LIMIT = float(backtest._cfg("MAX_DRAWDOWN_LIMIT", "30.0"))
LOCKBOX_MIN_BARS = int(os.environ.get("LOCKBOX_MIN_BARS", "150"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _ensure_promotions() -> None:
    if not PROMOTIONS.exists():
        PROMOTIONS.write_text(PROMOTIONS_HEADER, encoding="utf-8")


def _append_row(row: list[str]) -> None:
    _ensure_promotions()
    with PROMOTIONS.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def _lockbox_bar_count(symbols_spec: str) -> int:
    """Min post-warmup lockbox bar count across the basket. A thin symbol gates
    the whole promotion so we never PASS on a single overweight symbol."""
    import pandas as pd
    resolved = backtest._resolve_symbols(symbols_spec)
    counts: list[int] = []
    for _, path in resolved:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        s = backtest._slice(df, "lockbox")
        counts.append(len(s))
    return min(counts) if counts else 0


def has_promotion(commit_sha: str) -> bool:
    """Public helper used by live_trade.py to gate startup. Matches by 7-char
    short SHA prefix (the format git_short_sha() in loop.py uses)."""
    if not PROMOTIONS.exists():
        return False
    short = commit_sha[:7]
    for line in PROMOTIONS.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < len(PROMOTIONS_COLS):
            continue
        row = dict(zip(PROMOTIONS_COLS, parts))
        if row["commit"].startswith(short) and row["decision"] == "PASS":
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Promote current HEAD to live by lockbox check.")
    p.add_argument("--symbols", default=None,
                   help="comma-sep parquet stems under data/. Default: configs.toml symbols.")
    args = p.parse_args()

    head = _git("rev-parse", "--short=7", "HEAD")
    symbols_spec = args.symbols or backtest._cfg("SYMBOLS", "stocks/TSLA_1d,stocks/NVDA_1d,stocks/PYPL_1d")

    # Sample-size precheck — refuse if the lockbox window is too thin to draw
    # any conclusion from. Records the refusal so the audit trail still shows
    # the peek attempt.
    n_bars = _lockbox_bar_count(symbols_spec)
    if n_bars < LOCKBOX_MIN_BARS:
        reason = f"lockbox_bars={n_bars}<{LOCKBOX_MIN_BARS}"
        _append_row([
            _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            head, symbols_spec,
            "0.000000", "0.00", "0", str(n_bars),
            "REFUSED", reason,
        ])
        print(f"[promote] REFUSED: {reason}")
        print(f"[promote] wait for lockbox window to accrue more bars, or move lockbox_start back.")
        return 2

    print(f"[promote] commit={head}  symbols={symbols_spec}  lockbox_bars={n_bars}")
    print(f"[promote] running lockbox backtest…")
    metrics = backtest.run(symbols_spec, window="lockbox")

    sharpe = float(metrics.get("val_sharpe", 0.0))  # key name is legacy; this IS the lockbox sharpe when window=lockbox
    max_dd = float(metrics.get("max_drawdown", 0.0))
    trades = int(metrics.get("total_trades", 0))

    reasons: list[str] = []
    if sharpe <= 0:
        reasons.append(f"sharpe={sharpe:.4f}≤0")
    if max_dd >= MAX_DD_LIMIT:
        reasons.append(f"max_dd={max_dd:.2f}≥{MAX_DD_LIMIT}")
    if trades < MIN_TRADES:
        reasons.append(f"trades={trades}<{MIN_TRADES}")

    decision = "PASS" if not reasons else "FAIL"

    _append_row([
        _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        head, symbols_spec,
        f"{sharpe:.6f}", f"{max_dd:.2f}", str(trades), str(n_bars),
        decision, ";".join(reasons),
    ])

    print(f"[promote] sharpe={sharpe:.4f}  max_dd={max_dd:.2f}%  trades={trades}")
    print(f"[promote] decision: {decision}")
    if reasons:
        print(f"[promote] failed gates: {'; '.join(reasons)}")
    print(f"[promote] audit appended to {PROMOTIONS.relative_to(ROOT)}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
