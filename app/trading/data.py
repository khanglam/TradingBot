import asyncio
import pandas as pd
from tvdatafeed import TvDatafeed, Interval
from typing import Optional

class TradingViewDataFetcher:
    def __init__(self, username: str, password: str):
        self.tv = TvDatafeed(username, password)

    async def fetch_ohlcv(self, symbol: str, exchange: str, interval: str = '1h', n_bars: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data asynchronously from TradingView.
        Interval: '1m', '5m', '15m', '1h', '4h', '1d', etc.
        """
        # Map string interval to tvdatafeed Interval
        interval_map = {
            '1m': Interval.in_1_minute,
            '5m': Interval.in_5_minute,
            '15m': Interval.in_15_minute,
            '1h': Interval.in_1_hour,
            '4h': Interval.in_4_hour,
            '1d': Interval.in_daily
        }
        tv_interval = interval_map.get(interval, Interval.in_1_hour)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: self.tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
        )
        return df

    def fetch_ohlcv_sync(self, symbol: str, exchange: str, interval: str = '1h', n_bars: int = 500) -> pd.DataFrame:
        """Synchronous wrapper for compatibility."""
        interval_map = {
            '1m': Interval.in_1_minute,
            '5m': Interval.in_5_minute,
            '15m': Interval.in_15_minute,
            '1h': Interval.in_1_hour,
            '4h': Interval.in_4_hour,
            '1d': Interval.in_daily
        }
        tv_interval = interval_map.get(interval, Interval.in_1_hour)
        return self.tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
