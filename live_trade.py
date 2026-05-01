"""Paper / live trader — thin executor that ALWAYS uses the current strategy.py.

This is the deployment surface. The autoresearch loop NEVER touches this file —
it only optimizes strategy.py. Whatever signal logic strategy.py currently
encodes is what gets paper-traded, automatically.

## Design

Unlike the previous Lumibot-based implementation (which hard-coded an EMA
crossover that drifted from strategy.py the moment the loop changed it), this
executor reuses the exact same evaluation path as scan.py:

    1. Fetch recent bars (yfinance for stocks, alpaca-py CryptoHistoricalData
       for crypto — free, no API key needed for crypto bars).
    2. Run a backtesting.py Backtest using the in-tree `Strategy` class.
    3. Inspect `_trades`: if the most recent trade entered/exited within the
       last `--fresh` bars, that's a fresh BUY/SELL signal.
    4. Translate that signal into an Alpaca paper order via the modern
       `alpaca-py` SDK (https://github.com/alpacahq/alpaca-py):
         BUY  + flat            → market buy of  cash / last_close  units
         SELL + open position   → market sell of full position
         no signal              → no-op
    5. Idempotency: before submitting, check existing positions + open orders
       so re-runs in the same bar don't double-submit.
    6. Append a JSONL record to results/paper.log per symbol per run.

## Setup once

    1. Alpaca paper account (free): https://app.alpaca.markets/paper/dashboard/overview
    2. .env at project root:
           ALPACA_API_KEY=...
           ALPACA_API_SECRET=...
           ALPACA_PAPER=True            # False to use live keys
           PAPER_PER_SYMBOL_CASH=10000
           PAPER_WATCHLIST=SPY,QQQ,TSLA,NVDA
           PAPER_CRYPTO_WATCHLIST=BTC/USD,ETH/USD
    3. pip install alpaca-py (only required for non-dry runs).

## Usage

    python live_trade.py --dry --symbols SPY --asset stock     # no Alpaca needed
    python live_trade.py --asset stock                          # paper-trade stocks
    python live_trade.py --asset crypto                         # paper-trade crypto
    python live_trade.py --live --asset stock                   # REAL MONEY (careful)

This file is intentionally framework-light: no Lumibot, no broker abstraction,
just alpaca-py and backtesting.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
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
PAPER_LOG = RESULTS_DIR / "paper.log"

ALPACA_SIGNUP_URL = "https://app.alpaca.markets/paper/dashboard/overview"

DEFAULT_PER_SYMBOL_CASH = float(os.environ.get("PAPER_PER_SYMBOL_CASH", 10000))


# ─────────────────────────── data fetchers ────────────────────────────

def _fetch_stock_bars(symbol: str, days: int) -> pd.DataFrame:
    """Daily stock bars via yfinance (matches scan._fetch_daily exactly)."""
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


def _fetch_crypto_bars(symbol: str, days: int) -> pd.DataFrame:
    """Daily crypto bars via alpaca-py public CryptoHistoricalDataClient.

    No API key required for crypto market data, but we pass keys if available
    so the same client can be used in production. Returns a DataFrame with the
    backtesting.py column shape.
    """
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as e:
        raise SystemExit(
            "alpaca-py not installed. Run: pip install alpaca-py\n"
            f"(original error: {e})"
        )

    api_key = os.environ.get("ALPACA_API_KEY") or None
    api_secret = os.environ.get("ALPACA_API_SECRET") or None
    client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(days, 60) + 5)

    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(req)
    df = bars.df
    if df is None or df.empty:
        raise RuntimeError(f"alpaca-py returned no crypto bars for {symbol}")

    # alpaca-py returns a multiindex (symbol, timestamp). Flatten to timestamp index.
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)

    df = df.rename(
        columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def _fetch_bars(symbol: str, asset: str, days: int) -> pd.DataFrame:
    if asset == "crypto":
        return _fetch_crypto_bars(symbol, days)
    return _fetch_stock_bars(symbol, days)


# ─────────────────────────── signal extraction ──────────────────────────

def _scan_symbol(symbol: str, asset: str, days: int, fresh_bars: int, cash: float) -> dict:
    """Run strategy.py against recent bars and return the signal dict.

    Mirrors scan._scan_symbol so behaviour is identical across alerts and
    paper execution. Returned dict keys:
        symbol, signal ("BUY"/"SELL"/None), last_close, last_bar,
        entry_price/entry_time or exit_price/exit_time, error
    """
    out: dict = {"symbol": symbol, "signal": None, "last_bar": None}
    try:
        df = _fetch_bars(symbol, asset, days)
    except Exception as e:
        out["error"] = f"data fetch failed: {e}"
        return out

    if len(df) < 30:
        out["error"] = f"only {len(df)} bars; need at least 30"
        return out

    out["last_close"] = float(df["Close"].iloc[-1])
    out["last_bar"] = df.index[-1].isoformat()

    try:
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

    fresh_cutoff = df.index[-min(fresh_bars, len(df))]
    most_recent = trades.iloc[-1]
    entry_time = pd.Timestamp(most_recent["EntryTime"])
    exit_time = pd.Timestamp(most_recent["ExitTime"]) if pd.notna(most_recent.get("ExitTime")) else None

    if exit_time is not None and exit_time >= fresh_cutoff:
        out.update({
            "signal": "SELL",
            "exit_price": float(most_recent["ExitPrice"]),
            "exit_time": exit_time.isoformat(),
        })
        return out

    if exit_time is None and entry_time >= fresh_cutoff:
        out.update({
            "signal": "BUY",
            "entry_price": float(most_recent["EntryPrice"]),
            "entry_time": entry_time.isoformat(),
        })
        return out

    return out


# ─────────────────────────── Alpaca execution ───────────────────────────

def _alpaca_clients(paper: bool):
    """Lazy-import alpaca-py and return (trading_client, asset_class_helpers).

    Raises SystemExit with a friendly message if keys are missing or the
    package isn't installed.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError as e:
        raise SystemExit(
            "alpaca-py not installed. Run: pip install alpaca-py\n"
            f"(original error: {e})"
        )

    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_API_SECRET not set. Sign up for paper "
            f"keys at {ALPACA_SIGNUP_URL} and add them to .env."
        )

    client = TradingClient(api_key, api_secret, paper=paper)
    helpers = {
        "OrderSide": OrderSide,
        "TimeInForce": TimeInForce,
        "MarketOrderRequest": MarketOrderRequest,
        "GetOrdersRequest": GetOrdersRequest,
        "QueryOrderStatus": QueryOrderStatus,
    }
    return client, helpers


def _alpaca_symbol(symbol: str, asset: str) -> str:
    """Alpaca expects 'BTC/USD' for crypto and 'TSLA' for equities. Both are
    already in those forms — just normalize whitespace/case."""
    s = symbol.strip().upper()
    return s


def _get_position_qty(client, symbol: str) -> float:
    """Return open position quantity for symbol, or 0.0 if flat."""
    try:
        pos = client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        # alpaca-py raises APIError when no position exists — treat as flat
        return 0.0


def _has_open_order(client, helpers, symbol: str) -> bool:
    """True if any open order exists for symbol (avoid double-submit)."""
    req = helpers["GetOrdersRequest"](
        status=helpers["QueryOrderStatus"].OPEN,
        symbols=[symbol],
    )
    orders = client.get_orders(filter=req)
    return bool(orders)


def _submit_buy(client, helpers, symbol: str, asset: str, cash: float, last_close: float) -> dict:
    """Compute qty and submit a market buy. Crypto uses fractional qty,
    stocks use whole-share qty (more conservative + works with most paper
    accounts even if fractional is enabled)."""
    if last_close <= 0:
        return {"action": "skip", "reason": "non-positive last_close"}

    raw_qty = cash / last_close
    if asset == "crypto":
        qty = round(raw_qty, 6)  # 6 decimals plenty for BTC/ETH
    else:
        qty = float(int(raw_qty))  # whole shares only

    if qty <= 0:
        return {"action": "skip", "reason": f"qty<=0 (cash={cash}, px={last_close})"}

    tif = helpers["TimeInForce"].GTC if asset == "crypto" else helpers["TimeInForce"].DAY
    req = helpers["MarketOrderRequest"](
        symbol=symbol,
        qty=qty,
        side=helpers["OrderSide"].BUY,
        time_in_force=tif,
    )
    order = client.submit_order(req)
    return {
        "action": "buy",
        "qty": qty,
        "submitted_at": str(getattr(order, "submitted_at", "")),
        "order_id": str(getattr(order, "id", "")),
    }


def _submit_sell(client, helpers, symbol: str, asset: str, qty: float) -> dict:
    if qty <= 0:
        return {"action": "skip", "reason": "no position to close"}

    tif = helpers["TimeInForce"].GTC if asset == "crypto" else helpers["TimeInForce"].DAY
    req = helpers["MarketOrderRequest"](
        symbol=symbol,
        qty=qty,
        side=helpers["OrderSide"].SELL,
        time_in_force=tif,
    )
    order = client.submit_order(req)
    return {
        "action": "sell",
        "qty": qty,
        "submitted_at": str(getattr(order, "submitted_at", "")),
        "order_id": str(getattr(order, "id", "")),
    }


# ─────────────────────────── per-symbol dispatch ─────────────────────────

def _execute(scan: dict, asset: str, paper: bool, cash: float, dry: bool) -> dict:
    """Translate a scan result into Alpaca order action (or no-op).
    Returns an `execution` sub-dict to be merged into the JSONL log record."""
    symbol = _alpaca_symbol(scan["symbol"], asset)
    sig = scan.get("signal")
    last_close = scan.get("last_close")
    err = scan.get("error")

    if err:
        return {"action": "skip", "reason": f"scan error: {err}"}

    if sig not in ("BUY", "SELL"):
        return {"action": "noop", "reason": "no fresh signal"}

    if dry:
        return {
            "action": "dry-" + sig.lower(),
            "would_qty": (round(cash / last_close, 6) if asset == "crypto"
                          else (int(cash / last_close) if last_close else 0)) if sig == "BUY" else "full position",
            "reason": "--dry; not submitting",
        }

    client, helpers = _alpaca_clients(paper=paper)

    # Idempotency guard
    if _has_open_order(client, helpers, symbol):
        return {"action": "skip", "reason": "open order already exists for symbol"}

    qty_held = _get_position_qty(client, symbol)

    if sig == "BUY":
        if qty_held > 0:
            return {"action": "skip", "reason": f"already long {qty_held}"}
        return _submit_buy(client, helpers, symbol, asset, cash, last_close)

    # SELL
    if qty_held <= 0:
        return {"action": "skip", "reason": "no open position to sell"}
    return _submit_sell(client, helpers, symbol, asset, qty_held)


# ───────────────────────────── logging ───────────────────────────────

def _append_log(record: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with PAPER_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────── main ─────────────────────────────────

def _default_symbols(asset: str) -> str:
    if asset == "crypto":
        return os.environ.get("PAPER_CRYPTO_WATCHLIST", "BTC/USD,ETH/USD")
    return os.environ.get("PAPER_WATCHLIST", "SPY,QQQ,TSLA,NVDA")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=None,
                   help="Comma-sep symbols. Default: PAPER_WATCHLIST or PAPER_CRYPTO_WATCHLIST.")
    p.add_argument("--asset", choices=["stock", "crypto"], default="stock")
    p.add_argument("--days", type=int, default=int(os.environ.get("PAPER_LOOKBACK_DAYS", 250)))
    p.add_argument("--fresh", type=int, default=int(os.environ.get("PAPER_FRESH_BARS", 1)))
    p.add_argument("--cash", type=float, default=DEFAULT_PER_SYMBOL_CASH,
                   help="Per-symbol cash budget (default $10k, env PAPER_PER_SYMBOL_CASH).")
    p.add_argument("--dry", action="store_true",
                   help="Compute signals only — do NOT submit orders or import alpaca trading.")
    p.add_argument("--live", action="store_true",
                   help="Use Alpaca LIVE keys (real money). Default: paper.")
    args = p.parse_args()

    symbols_str = args.symbols or _default_symbols(args.asset)
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    if not symbols:
        print("ERROR: no symbols (set PAPER_WATCHLIST/PAPER_CRYPTO_WATCHLIST or pass --symbols).",
              file=sys.stderr)
        return 2

    paper_env = os.environ.get("ALPACA_PAPER", "True").lower() != "false"
    paper = paper_env and not args.live

    mode = "DRY" if args.dry else ("PAPER" if paper else "LIVE")
    print(f"[paper] mode={mode}  asset={args.asset}  symbols={','.join(symbols)}")
    print(f"[paper] strategy: {(ROOT / 'strategy.py').read_text(encoding='utf-8').splitlines()[0]}")
    print(f"[paper] per-symbol cash=${args.cash:,.0f}  lookback={args.days}d  fresh={args.fresh}")

    # If we're going to actually submit, fail fast on missing creds — the user
    # has explicit instructions: no fallbacks, only real data.
    if not args.dry:
        if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
            print(
                f"\nERROR: ALPACA_API_KEY / ALPACA_API_SECRET not set in environment.\n"
                f"Sign up for free paper keys: {ALPACA_SIGNUP_URL}\n"
                f"Then add them to .env (see .env.example).\n"
                f"Or pass --dry to compute signals without submitting orders.",
                file=sys.stderr,
            )
            return 3

    fired_any = False
    for sym in symbols:
        scan = _scan_symbol(sym, args.asset, days=args.days, fresh_bars=args.fresh, cash=args.cash)
        sig = scan.get("signal") or "—"
        err = scan.get("error", "")
        print(f"[paper] {sym}: {sig} {err}".rstrip())

        try:
            execution = _execute(scan, asset=args.asset, paper=paper, cash=args.cash, dry=args.dry)
        except SystemExit:
            raise
        except Exception as e:
            execution = {"action": "error", "reason": f"{e}\n{traceback.format_exc()}"}

        print(f"[paper] {sym}: -> {execution.get('action')} ({execution.get('reason', '')})".rstrip())
        if execution.get("action") not in ("noop", "skip", "error"):
            fired_any = True

        _append_log({
            "mode": mode,
            "asset": args.asset,
            "symbol": sym,
            "scan": scan,
            "execution": execution,
        })

    return 0 if fired_any else 1  # exit 1 = no actionable signals (CI conditional friendly)


if __name__ == "__main__":
    sys.exit(main())
