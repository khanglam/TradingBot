"""Crypto campaign strategy. Mutated by the autoresearch loop when
STRATEGY_FILE points here (default for the crypto matrix shard).

Baseline: Donchian-breakout trend-following (Turtle System One on 4h bars).

Why this and not the stocks EMA crossover:
  - Crypto bars are 4h not 1d, so 24/7 markets, 6 bars/day, ~2190 bars/year.
  - Crypto trends are sharper and more sustained than stock trends; breakout
    systems consistently beat EMA crossovers on volatile commodities. The
    Turtle Traders proved this on cocoa, sugar, and yen — same shape of
    distribution as BTC/ETH (fat tails, persistent trends, sharp reversals).
  - Two exits in parallel — short-Donchian band break (recent weakness) AND
    ATR trailing stop (volatility-aware hard stop) — handle the two ways a
    crypto trend dies: slow rollover vs flash crash.
  - 20/15-bar Donchian (~3.3-day entry, ~2.5-day exit) trades often enough
    to clear MIN_TRADES on a 2-year val window without becoming pure noise.

The agent is free to replace any of this. This is a starting point with
real defensible reasoning.
"""

import numpy as np
import pandas as pd
from backtesting import Strategy as _BTStrategy

# Minimum bars live_trade.py must fetch before evaluating this strategy.
# Covers ATR(14) chained into ATR-MA(50) plus Donchian(24) × ~3 for full
# warm-up. Keep this defined at module scope — backtest.strategy_min_bars()
# reads it to size the live bar window so live indicator state matches
# backtest. Falls back to 200 if removed.
MIN_BARS_REQUIRED = 250


def _donchian(high: pd.Series, low: pd.Series, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Donchian channels: rolling N-bar high and N-bar low.
    Returns (upper, lower). Classic Turtle entry: close > upper(N).
    Classic Turtle exit: close < lower(M) where M < N."""
    upper = pd.Series(high).rolling(n).max().to_numpy().copy()
    lower = pd.Series(low).rolling(n).min().to_numpy().copy()
    return upper, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> np.ndarray:
    """Average True Range — volatility measure for stop sizing.
    On 4h BTC bars typical ATR is 1-2% of price in calm regimes,
    3-5% in trend, 8%+ in panic."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill().to_numpy().copy()


def _atr_ma(high: pd.Series, low: pd.Series, close: pd.Series, atr_n: int = 14, ma_n: int = 50) -> np.ndarray:
    """50-bar moving average of ATR. Used to compute volatility regime ratio."""
    atr = _atr(high, low, close, atr_n)
    return pd.Series(atr).rolling(ma_n).mean().bfill().to_numpy().copy()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> np.ndarray:
    """Average Directional Index — measures trend strength.
    Returns ADX values. ADX > 25 indicates a strong trend (directional movement
    is well-established). Used as momentum confirmation on entry: only take
    breakouts when ADX confirms the market is trending, not ranging."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr = _atr(high, low, close, n)
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    return adx.fillna(0).bfill().to_numpy().copy()


def _di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Returns (plus_di, minus_di) — Directional Indicators.
    Plus_DI > Minus_DI indicates bullish momentum; opposite for bearish.
    Used to filter only breakouts that align with the dominant direction."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr = _atr(high, low, close, n)
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    return plus_di.fillna(50).to_numpy().copy(), minus_di.fillna(50).to_numpy().copy()


def _sma(series: pd.Series, n: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(series).rolling(n).mean().to_numpy().copy()


class Strategy(_BTStrategy):
    # Turtle System One: 28-bar breakout entry, 15-bar opposite exit.
    # On 4h bars this is ~4.7-day entry confirmation, ~2.5-day exit signal.
    breakout_period = 28
    exit_period = 15

    # ATR trailing stop — volatility-aware hard exit. Now regime-adaptive AND
    # time-adaptive: multiplier increases with bars in position (up to +1.0
    # at time_exit_bars). This lets strong trends run longer before being
    # stopped out, reducing premature exits on early pullbacks.
    atr_period = 14
    atr_multiplier_low_vol = 3.0   # tight stop in calm markets (increased from 2.5)
    atr_multiplier_high_vol = 3.6  # loose stop in trending volatility
    atr_time_extra = 1.0           # max extra multiplier after time_exit_bars

    # Volatility-adaptive entry gate: only breakout when current ATR is
    # above 0.95x the 50-bar moving average of ATR. Filters out breakfakes
    # in ranging regimes where volatility is suppressed and price motion
    # lacks persistence.
    atr_ma_period = 50
    atr_vol_threshold = 0.95

    # ADX momentum confirmation: require ADX > 25 on entry bars to confirm
    # the market is trending. ADX is a momentum indicator that filters at
    # the bar level without blocking regime windows, strengthening the entry
    # signal by requiring trend strength, not trend direction.
    adx_period = 14
    adx_threshold = 25.0

    # Volume confirmation: require current bar volume > 1.2x its 20-bar SMA
    # to confirm the breakout is backed by real participation, not a thin-air
    # spike.
    vol_sma_period = 20
    vol_threshold = 1.2

    # Long-term trend filter: close must be above 200-period SMA.
    # Ensures we only take long breakouts in a confirmed uptrend regime.
    sma_period = 200

    # Volatility-scaled position sizing: size INVERSELY proportional to
    # realized volatility. High ATR → small size, low ATR → larger size.
    vol_scale_floor = 0.20
    vol_scale_ceil = 1.05
    base_fraction = 0.55

    # Time-decay exit: exit after N bars in position, regardless of P&L.
    # 40 bars ≈ 6-7 days on 4h chart.
    time_exit_bars = 40

    def init(self) -> None:
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        self.upper, _ = self.I(_donchian, high, low, self.breakout_period)
        self.atr = self.I(_atr, high, low, close, self.atr_period)
        self.atr_ma = self.I(_atr_ma, high, low, close, self.atr_period, self.atr_ma_period)
        self.adx = self.I(_adx, high, low, close, self.adx_period)
        self.plus_di, self.minus_di = self.I(_di, high, low, close, self.adx_period)
        self.vol_sma = self.I(_sma, self.data.Volume, self.vol_sma_period)
        self.sma200 = self.I(_sma, close, self.sma_period)

        self.highest_price: float | None = None
        self.entry_bar: int | None = None

    def next(self) -> None:
        if len(self.data) < self.breakout_period + 1:
            return

        close = self.data.Close[-1]
        current_bar = len(self.data) - 1

        # Entry conditions
        breakout = close > self.upper[-2]
        vol_regime_high = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
        momentum_confirm = self.adx[-1] > self.adx_threshold
        volume_confirm = self.data.Volume[-1] > self.vol_sma[-1] * self.vol_threshold
        trend_filter = close > self.sma200[-1]

        if breakout and vol_regime_high and momentum_confirm and volume_confirm and trend_filter and not self.position:
            # Inverse volatility scaling
            vol_ratio = self.atr_ma[-1] / max(self.atr[-1], 0.0001)
            size_multiplier = max(self.vol_scale_floor, min(self.vol_scale_ceil, vol_ratio))
            size = self.base_fraction * size_multiplier
            self.buy(size=size)
            self.highest_price = close
            self.entry_bar = current_bar
        elif self.position:
            # Update running peak
            self.highest_price = max(self.highest_price, close)
            bars_in_position = current_bar - self.entry_bar

            # Regime-adaptive base multiplier
            is_high_vol = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
            base_mult = self.atr_multiplier_high_vol if is_high_vol else self.atr_multiplier_low_vol

            # Time-adaptive increment: multiplier grows with bars in position
            # up to max of +atr_time_extra at time_exit_bars.
            time_factor = min(1.0, bars_in_position / self.time_exit_bars) * self.atr_time_extra
            atr_mult = base_mult + time_factor
            trailing_stop = self.highest_price - self.atr[-1] * atr_mult
            stop_hit = close <= trailing_stop

            # Time-decay exit
            time_exit_hit = bars_in_position >= self.time_exit_bars

            if stop_hit or time_exit_hit:
                self.position.close()
                self.highest_price = None
                self.entry_bar = None