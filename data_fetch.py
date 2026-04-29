"""Download OHLCV data and cache as Parquet.

Crypto via CCXT (Binance public API, free, no key required).
Stocks via yfinance (free, scraped from Yahoo).

Output layout:
    data/crypto/{SYMBOL}_{TIMEFRAME}.parquet   e.g. data/crypto/BTC_USDT_4h.parquet
    data/stocks/{SYMBOL}_{INTERVAL}.parquet    e.g. data/stocks/SPY_1d.parquet

Each Parquet file has columns: Open, High, Low, Close, Volume
indexed by a timezone-naive UTC DatetimeIndex (backtesting.py expects this).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def _bn_timeframe_ms(tf: str) -> int:
    n, unit = int(tf[:-1]), tf[-1]
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


def fetch_crypto(
    symbol: str = "BTC/USDT",
    exchange: str = "kucoin",
    timeframe: str = "4h",
    start: str = "2019-01-01",
    end: str | None = None,
) -> Path:
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end or "now", tz="UTC").timestamp() * 1000)
    step = _bn_timeframe_ms(timeframe)

    rows: list[list] = []
    cursor = since
    while cursor < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
        time.sleep(ex.rateLimit / 1000)
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["ts"])

    out = DATA_DIR / "crypto" / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[crypto] {symbol} {timeframe}  {len(df):,} rows  {df.index.min()} -> {df.index.max()}")
    print(f"         saved to {out}")
    return out


def fetch_stocks(
    symbol: str = "SPY",
    interval: str = "1d",
    start: str = "2019-01-01",
    end: str | None = None,
) -> Path:
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)

    out = DATA_DIR / "stocks" / f"{symbol}_{interval}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[stocks] {symbol} {interval}  {len(df):,} rows  {df.index.min()} -> {df.index.max()}")
    print(f"         saved to {out}")
    return out


def load(path: str | Path) -> pd.DataFrame:
    """Load a cached Parquet file as the OHLCV DataFrame the backtest expects."""
    return pd.read_parquet(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asset", choices=["crypto", "stocks", "all"], default="all")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None, help="crypto: 1m/5m/1h/4h/1d  stocks: 1d/1h/15m")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    if args.asset in ("crypto", "all"):
        fetch_crypto(
            symbol=args.symbol or "BTC/USDT",
            timeframe=args.timeframe or "4h",
            start=args.start,
            end=args.end,
        )
    if args.asset in ("stocks", "all"):
        fetch_stocks(
            symbol=args.symbol or "SPY",
            interval=args.timeframe or "1d",
            start=args.start,
            end=args.end,
        )
