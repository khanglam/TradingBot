"""Download OHLCV data and cache as Parquet.

Crypto via CCXT (KuCoin public API, free, no key required).
Stocks via Alpaca Market Data API (alpaca-py, IEX feed on free tier).

Output layout:
    data/crypto/{SYMBOL}_{TIMEFRAME}.parquet   e.g. data/crypto/BTC_USDT_4h.parquet
    data/stocks/{SYMBOL}_{INTERVAL}.parquet    e.g. data/stocks/TSLA_1h.parquet

Each Parquet file has columns: Open, High, Low, Close, Volume
indexed by a timezone-naive UTC DatetimeIndex (backtesting.py expects this).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_DIR = Path(__file__).parent / "data"


def _bn_timeframe_ms(tf: str) -> int:
    n, unit = int(tf[:-1]), tf[-1]
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


def parse_interval(interval: str):
    """Map '1h', '1d', '15m' to an alpaca-py TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf_str = interval.strip().lower()
    m = re.fullmatch(r"(\d+)\s*(m|min|minute|h|hr|hour|d|day)", tf_str)
    if not m:
        raise ValueError(
            f"Unrecognised interval {interval!r}. Use e.g. '1h', '15m', '1d'."
        )
    amount = int(m.group(1))
    unit_str = m.group(2)
    if unit_str in ("m", "min", "minute"):
        unit = TimeFrameUnit.Minute
    elif unit_str in ("h", "hr", "hour"):
        unit = TimeFrameUnit.Hour
    else:
        unit = TimeFrameUnit.Day
    return TimeFrame(amount, unit)


def normalize_alpaca_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten alpaca-py bar MultiIndex and return backtesting.py OHLCV shape."""
    if df is None or df.empty:
        raise RuntimeError(f"Alpaca returned no rows for {symbol}")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out


def _alpaca_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env for stock data. "
            "Get free paper keys at https://app.alpaca.markets/paper/dashboard/overview"
        )
    return api_key, api_secret


def _stock_feed():
    from alpaca.data.enums import DataFeed

    name = os.environ.get("ALPACA_STOCK_FEED", "iex").strip().lower()
    if name == "sip":
        return DataFeed.SIP
    return DataFeed.IEX


def _to_utc_ts(value: str | datetime | pd.Timestamp | None, default: str = "2018-01-01") -> pd.Timestamp:
    ts = pd.Timestamp(value if value is not None else default)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _chunk_days(interval: str) -> int:
    """Calendar days per Alpaca request chunk (stay under ~10k bars)."""
    tf = interval.strip().lower()
    if tf.endswith(("m", "min", "minute")):
        return 14
    if tf.endswith(("h", "hr", "hour")):
        return 90
    return 365


def fetch_stock_bars(
    symbol: str,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pd.DataFrame:
    """Fetch stock OHLCV from Alpaca (not cached). Used by live_trade and data_fetch."""
    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    api_key, api_secret = _alpaca_credentials()
    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    timeframe = parse_interval(interval)
    feed = _stock_feed()

    start_ts = _to_utc_ts(start, default="2018-01-01")
    end_ts = _to_utc_ts(end, default="now")

    chunk = pd.Timedelta(days=_chunk_days(interval))
    frames: list[pd.DataFrame] = []
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + chunk, end_ts)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=cursor.to_pydatetime(),
            end=chunk_end.to_pydatetime(),
            feed=feed,
            adjustment=Adjustment.ALL,
        )
        bars = client.get_stock_bars(req)
        part = bars.df
        if part is not None and not part.empty:
            frames.append(normalize_alpaca_bars(part, symbol))
        cursor = chunk_end
        if cursor < end_ts:
            time.sleep(0.25)

    if not frames:
        raise RuntimeError(f"Alpaca returned no rows for {symbol} {interval}")

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


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
    interval: str = "1h",
    start: str = "2018-01-01",
    end: str | None = None,
) -> Path:
    df = fetch_stock_bars(symbol=symbol, interval=interval, start=start, end=end)
    out = DATA_DIR / "stocks" / f"{symbol}_{interval}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[stocks] {symbol} {interval}  {len(df):,} rows  {df.index.min()} -> {df.index.max()}")
    print(f"         saved to {out}")
    return out


def fetch_if_missing(path: str | Path, start: str | None = None, *, force: bool = False) -> Path:
    """Make sure a data/<asset>/<symbol>_<tf>.parquet file is on disk; fetch if not.

    Idempotent — returns immediately when the file already exists (unless
    `force=True`). Otherwise parses asset class / symbol / timeframe from the
    path layout and delegates to `fetch_crypto` or `fetch_stocks`.
    """
    path = Path(path)
    if path.exists() and not force:
        return path
    if path.exists() and force:
        path.unlink()

    try:
        rel = path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise ValueError(f"path must be under {DATA_DIR}, got {path}")

    parts = rel.parts
    if len(parts) != 2:
        raise ValueError(f"expected data/<asset>/<symbol>_<tf>.parquet, got {rel}")
    asset = parts[0]
    stem = Path(parts[1]).stem

    bits = stem.rsplit("_", 1)
    if len(bits) != 2:
        raise ValueError(f"can't infer timeframe from {stem!r}; expected '<symbol>_<tf>'")
    symbol_us, timeframe = bits

    print(f"[data_fetch] missing {rel}; fetching {asset} {symbol_us} {timeframe}…", flush=True)

    if asset == "crypto":
        symbol = symbol_us.replace("_", "/")
        return fetch_crypto(symbol=symbol, timeframe=timeframe, start=start or "2019-01-01")
    if asset == "stocks":
        return fetch_stocks(symbol=symbol_us, interval=timeframe, start=start or "2018-01-01")
    raise ValueError(f"unknown asset class {asset!r}; expected 'crypto' or 'stocks'")


def timeframe_from_symbols_spec(spec: str, default: str = "1h") -> str:
    """Parse timeframe from e.g. 'stocks/TSLA_1h,stocks/NVDA_1h' -> '1h'."""
    entry = spec.split(",")[0].strip()
    if not entry:
        return default
    stem = entry.split("/")[-1]
    if stem.endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    bits = stem.rsplit("_", 1)
    if len(bits) == 2:
        return bits[1]
    return default


def load(path: str | Path) -> pd.DataFrame:
    """Load a cached Parquet file as the OHLCV DataFrame the backtest expects."""
    return pd.read_parquet(path)


def ensure_symbols_spec(spec: str, start: str | None = None, *, force: bool = False) -> None:
    """Ensure every parquet in a SYMBOLS spec exists under data/; fetch on miss."""
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        rel = entry[len("data/") :] if entry.startswith("data/") else entry
        if rel.endswith(".parquet"):
            rel = rel[: -len(".parquet")]
        fetch_if_missing(DATA_DIR / f"{rel}.parquet", start=start, force=force)


def fetch_campaign_symbols(*, force: bool = False) -> None:
    """Fetch all symbols from configs.toml for the active CAMPAIGN (no-op if cached)."""
    from backtest import _load_campaign_config

    cfg = _load_campaign_config()
    spec = os.environ.get("SYMBOLS") or cfg.get("symbols")
    if not spec:
        return
    start = cfg.get("data_fetch_start")
    ensure_symbols_spec(str(spec), start=str(start) if start is not None else None, force=force)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
        root = Path(__file__).parent
        for p in [root, *root.parents]:
            if (p / "configs.toml").exists() and (p / ".env").exists():
                load_dotenv(p / ".env", override=False)
                break
    except ImportError:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--asset", choices=["crypto", "stocks", "all"], default="all")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None, help="crypto: 1m/5m/1h/4h/1d  stocks: 1h/1d/15m")
    p.add_argument("--start", default=None, help="default: configs.toml data_fetch_start or 2019/2018")
    p.add_argument("--end", default=None)
    p.add_argument("--force", action="store_true", help="refetch even if parquet exists")
    args = p.parse_args()

    if args.symbol is None and os.environ.get("CAMPAIGN"):
        fetch_campaign_symbols(force=args.force)
    else:
        start = args.start
        if args.asset in ("crypto", "all"):
            out = DATA_DIR / "crypto" / f"{(args.symbol or 'BTC/USDT').replace('/', '_')}_{args.timeframe or '4h'}.parquet"
            if args.force and out.exists():
                out.unlink()
            fetch_crypto(
                symbol=args.symbol or "BTC/USDT",
                timeframe=args.timeframe or "4h",
                start=start or "2019-01-01",
                end=args.end,
            )
        if args.asset in ("stocks", "all"):
            sym = args.symbol or "SPY"
            tf = args.timeframe or "1h"
            out = DATA_DIR / "stocks" / f"{sym}_{tf}.parquet"
            if args.force and out.exists():
                out.unlink()
            fetch_stocks(
                symbol=sym,
                interval=tf,
                start=start or "2018-01-01",
                end=args.end,
            )
