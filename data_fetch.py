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
import os
import sys
import time
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


def fetch_crypto(
    symbol: str = "BTC/USDT",
    exchange: str = "kraken",
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


def fetch_if_missing(path: str | Path, start: str | None = None) -> Path:
    """Make sure a data/<asset>/<symbol>_<tf>.parquet file is on disk; fetch if not.

    Idempotent — returns immediately when the file already exists. Otherwise
    parses asset class / symbol / timeframe from the path layout and delegates
    to `fetch_crypto` or `fetch_stocks`. Blocks the caller for ~30–60s on the
    very first miss; subsequent calls are instant.

    Used by every code path that reads a backtest parquet (app.py endpoints,
    backtest.py:_run_single) so that a fresh checkout never crashes with a
    bare FileNotFoundError on missing OHLCV.

    `start` overrides the default fetch window (2019 for crypto, 2018 for stocks);
    pass it through if a particular caller wants a deeper history.
    """
    path = Path(path)
    if path.exists():
        return path

    try:
        rel = path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise ValueError(f"path must be under {DATA_DIR}, got {path}")

    parts = rel.parts
    if len(parts) != 2:
        raise ValueError(f"expected data/<asset>/<symbol>_<tf>.parquet, got {rel}")
    asset = parts[0]
    stem = Path(parts[1]).stem  # 'BTC_USDT_4h'

    bits = stem.rsplit("_", 1)
    if len(bits) != 2:
        raise ValueError(f"can't infer timeframe from {stem!r}; expected '<symbol>_<tf>'")
    symbol_us, timeframe = bits

    print(f"[data_fetch] missing {rel}; fetching {asset} {symbol_us} {timeframe} (~30–60s)…", flush=True)

    if asset == "crypto":
        symbol = symbol_us.replace("_", "/")
        return fetch_crypto(symbol=symbol, timeframe=timeframe, start=start or "2019-01-01")
    if asset == "stocks":
        return fetch_stocks(symbol=symbol_us, interval=timeframe, start=start or "2018-01-01")
    raise ValueError(f"unknown asset class {asset!r}; expected 'crypto' or 'stocks'")


def load(path: str | Path) -> pd.DataFrame:
    """Load a cached Parquet file as the OHLCV DataFrame the backtest expects."""
    return pd.read_parquet(path)


def _load_campaign_config() -> dict:
    import subprocess
    import tomllib

    campaign = os.environ.get("CAMPAIGN", "").strip()
    if not campaign:
        return {}
    root = Path(__file__).parent
    r = subprocess.run(
        ["git", "show", "origin/main:configs.toml"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if r.returncode == 0:
        cfg_text = r.stdout
    elif (root / "configs.toml").exists():
        cfg_text = (root / "configs.toml").read_text(encoding="utf-8")
    else:
        return {}
    all_campaigns = tomllib.loads(cfg_text)
    return all_campaigns.get(campaign, {})


def ensure_symbols_spec(spec: str, start: str | None = None) -> None:
    """Ensure every parquet in a SYMBOLS spec exists under data/; fetch on miss."""
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        rel = entry[len("data/") :] if entry.startswith("data/") else entry
        if rel.endswith(".parquet"):
            rel = rel[: -len(".parquet")]
        fetch_if_missing(DATA_DIR / f"{rel}.parquet", start=start)


def fetch_campaign_symbols() -> None:
    """Fetch all symbols from configs.toml for the active CAMPAIGN (no-op if cached)."""
    cfg = _load_campaign_config()
    spec = os.environ.get("SYMBOLS") or cfg.get("symbols")
    if not spec:
        return
    start = cfg.get("data_fetch_start")
    ensure_symbols_spec(str(spec), start=str(start) if start is not None else None)


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
    p.add_argument("--timeframe", default=None, help="crypto: 1m/5m/1h/4h/1d  stocks: 1d/1h/15m")
    p.add_argument("--start", default=None, help="default: configs.toml data_fetch_start or 2019/2018")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    if args.symbol is None and os.environ.get("CAMPAIGN"):
        fetch_campaign_symbols()
    else:
        start = args.start
        if args.asset in ("crypto", "all"):
            fetch_crypto(
                symbol=args.symbol or "BTC/USDT",
                timeframe=args.timeframe or "4h",
                start=start or "2019-01-01",
                end=args.end,
            )
        if args.asset in ("stocks", "all"):
            fetch_stocks(
                symbol=args.symbol or "SPY",
                interval=args.timeframe or "1d",
                start=start or "2018-01-01",
                end=args.end,
            )
