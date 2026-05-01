"""Watchlist scanner — runs strategy.py against a list of symbols, emits
BUY/SELL alerts when the strategy entered or exited a position on the most
recent bar.

This is the deployment surface for the "alert me, I execute manually"
workflow. It is the autoresearch loop's complement: the loop optimizes
strategy.py offline, the scanner uses whatever strategy.py currently
contains to scan markets and emit signals.

Scheduled to run via .github/workflows/scan.yml (daily, post-close PST).

Configuration via .env:
    SCAN_WATCHLIST           comma-sep stock symbols (default: SPY,QQQ)
    SCAN_LOOKBACK_DAYS       how much daily history to feed the strategy (default 250)
    SCAN_FRESH_BARS          a signal is "fresh" if it fired within the last N
                             bars (default 1 — i.e. only alert on today's bar)
    SCAN_WEBHOOK_URL         optional Discord/Slack-compatible webhook URL.
                             If unset, alerts only go to scan.log + stdout.
    SCAN_INITIAL_CASH        cash for the per-symbol backtest (default 10000)

Usage:
    python scan.py                                    # use SCAN_WATCHLIST
    python scan.py --symbols TSLA,NVDA --webhook ...
    python scan.py --dry                              # never POST, just print

Output: writes results/scan.log with a timestamped record of every scan and
which signals fired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SCAN_LOG = RESULTS_DIR / "scan.log"


# ─────────────────────────── data + backtest ────────────────────────────

def _fetch_daily(symbol: str, days: int) -> pd.DataFrame:
    """Pull recent daily bars from yfinance. Returns a DataFrame with the
    columns backtesting.py expects, or raises on failure."""
    import yfinance as yf
    df = yf.download(
        symbol,
        period=f"{max(days, 60)}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def _scan_symbol(symbol: str, days: int, fresh_bars: int, cash: float) -> dict:
    """Run strategy.py against recent bars for one symbol, return signal info.

    Returns a dict with: symbol, last_close, last_bar, signal ("BUY"/"SELL"/None),
    entry_price/exit_price (when applicable), and an `error` field on failure."""
    out: dict = {"symbol": symbol, "signal": None, "last_bar": None}
    try:
        df = _fetch_daily(symbol, days)
    except Exception as e:
        out["error"] = f"data fetch failed: {e}"
        return out

    if len(df) < 30:
        out["error"] = f"only {len(df)} bars; need at least 30"
        return out

    out["last_close"] = float(df["Close"].iloc[-1])
    out["last_bar"] = df.index[-1].isoformat()

    try:
        # Lazy imports — don't pay the cost if data fetch failed
        from backtesting import Backtest
        from strategy import Strategy as UserStrategy
        bt = Backtest(df, UserStrategy, cash=cash, commission=0.0,
                      exclusive_orders=True, finalize_trades=True)
        stats = bt.run()
    except Exception as e:
        out["error"] = f"backtest failed: {e}\n{traceback.format_exc()}"
        return out

    trades = stats.get("_trades")
    if trades is None or len(trades) == 0:
        return out

    # Compare timestamps as numpy datetime64 to avoid timezone hiccups.
    fresh_cutoff = df.index[-min(fresh_bars, len(df))]

    most_recent = trades.iloc[-1]
    entry_time = pd.Timestamp(most_recent["EntryTime"])
    exit_time = pd.Timestamp(most_recent["ExitTime"]) if pd.notna(most_recent.get("ExitTime")) else None

    # SELL signal: the most recent trade closed within fresh_bars.
    if exit_time is not None and exit_time >= fresh_cutoff:
        out.update({
            "signal": "SELL",
            "exit_price": float(most_recent["ExitPrice"]),
            "exit_time": exit_time.isoformat(),
        })
        return out

    # BUY signal: the most recent trade is still OPEN and entered within fresh_bars.
    # In backtesting.py's _trades, an unclosed trade has ExitTime == NaT (or ExitBar == -1).
    if exit_time is None and entry_time >= fresh_cutoff:
        out.update({
            "signal": "BUY",
            "entry_price": float(most_recent["EntryPrice"]),
            "entry_time": entry_time.isoformat(),
        })
        return out

    return out


# ───────────────────────────── alerts ───────────────────────────────────

def _format_alert(scans: list[dict]) -> str:
    """Human-readable summary of all scans, with BUY/SELL prominently flagged."""
    fired = [s for s in scans if s.get("signal") in ("BUY", "SELL")]
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**Scan {when}**  (watchlist: {len(scans)} symbols)"]
    if not fired:
        lines.append("_no fresh signals_")
    for s in fired:
        sig = s["signal"]
        emoji = "🟢" if sig == "BUY" else "🔴"
        if sig == "BUY":
            lines.append(
                f"{emoji} **{sig} {s['symbol']}** entry@${s['entry_price']:.2f} "
                f"(close ${s['last_close']:.2f})"
            )
        else:
            lines.append(
                f"{emoji} **{sig} {s['symbol']}** exit@${s['exit_price']:.2f} "
                f"(close ${s['last_close']:.2f})"
            )
    errors = [s for s in scans if s.get("error")]
    if errors:
        lines.append(f"_(skipped: {', '.join(s['symbol'] for s in errors)})_")
    return "\n".join(lines)


def _post_webhook(url: str, message: str) -> bool:
    """POST a Discord/Slack-compatible payload. Returns True on 2xx."""
    import urllib.request
    payload = json.dumps({"content": message, "text": message}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"[scan] webhook error: {e}", file=sys.stderr)
        return False


def _append_log(scans: list[dict], message: str, posted: bool) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "scans": scans,
        "message": message,
        "webhook_posted": posted,
    }
    with SCAN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────── main ─────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=os.environ.get("SCAN_WATCHLIST", "SPY,QQQ"))
    p.add_argument("--days", type=int, default=int(os.environ.get("SCAN_LOOKBACK_DAYS", 250)))
    p.add_argument("--fresh", type=int, default=int(os.environ.get("SCAN_FRESH_BARS", 1)))
    p.add_argument("--cash", type=float, default=float(os.environ.get("SCAN_INITIAL_CASH", 10000)))
    p.add_argument("--webhook", default=os.environ.get("SCAN_WEBHOOK_URL", ""))
    p.add_argument("--dry", action="store_true", help="never POST to webhook")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("ERROR: no symbols to scan (set SCAN_WATCHLIST or pass --symbols)", file=sys.stderr)
        return 2

    print(f"[scan] {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"[scan] strategy: {(ROOT / 'strategy.py').read_text(encoding='utf-8').splitlines()[0]}")

    scans: list[dict] = []
    for sym in symbols:
        result = _scan_symbol(sym, days=args.days, fresh_bars=args.fresh, cash=args.cash)
        sig = result.get("signal") or "—"
        err = result.get("error", "")
        print(f"[scan] {sym}: {sig} {err}".rstrip())
        scans.append(result)

    message = _format_alert(scans)
    print()
    print(message)
    print()

    fired = any(s.get("signal") in ("BUY", "SELL") for s in scans)
    posted = False
    if fired and args.webhook and not args.dry:
        posted = _post_webhook(args.webhook, message)
        print(f"[scan] webhook posted: {posted}")
    elif fired and args.dry:
        print("[scan] --dry set; not posting webhook")
    elif fired:
        print("[scan] no SCAN_WEBHOOK_URL configured; alert printed only")

    _append_log(scans, message, posted)
    return 0 if fired else 1  # exit 1 means "no signals" (useful for CI conditional steps)


if __name__ == "__main__":
    sys.exit(main())
