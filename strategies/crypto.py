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
# warm-up. Keep this at module scope — backtest.strategy_min_bars()
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

    # ATR trailing stop — volatility-aware hard exit. Regime-adaptive:
    # multiplier tightens in low vol, loosens in high vol.
    atr_period = 14
    atr_multiplier_low_vol = 3.0   # tight stop in calm markets
    atr_multiplier_high_vol = 3.6  # loose stop in trending volatility

    # Profit-dependent trailing stop: once profit reaches this many ATR,
    # tighten the trailing multiplier to lock in gains.
    profit_atr_threshold = 2.0
    tight_atr_multiplier = 2.0

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

    # Long-term trend filter: close must be above 200-period SMA.
    # Ensures we only take long breakouts in a confirmed uptrend regime.
    sma_period = 200

    # Volatility-scaled position sizing: size INVERSELY proportional to
    # realized volatility. High ATR → small size, low ATR → larger size.
    vol_scale_floor = 0.20
    vol_scale_ceil = 1.05
    base_fraction = 0.55

    def init(self) -> None:
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        self.upper, _ = self.I(_donchian, high, low, self.breakout_period)
        # Exit Donchian: lower channel of exit_period
        _, self.exit_lower = self.I(_donchian, high, low, self.exit_period)
        self.atr = self.I(_atr, high, low, close, self.atr_period)
        self.atr_ma = self.I(_atr_ma, high, low, close, self.atr_period, self.atr_ma_period)
        self.adx = self.I(_adx, high, low, close, self.adx_period)
        self.plus_di, self.minus_di = self.I(_di, high, low, close, self.adx_period)
        self.sma200 = self.I(_sma, close, self.sma_period)

        self.highest_price: float | None = None
        self.entry_price: float | None = None
        self.entry_bar: int | None = None

    def next(self) -> None:
        if len(self.data) < self.breakout_period + 1:
            return

        close = self.data.Close[-1]
        current_bar = len(self.data) - 1

        # Entry conditions (volume confirmation removed)
        breakout = close > self.upper[-2]
        vol_regime_high = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
        momentum_confirm = self.adx[-1] > self.adx_threshold
        trend_filter = close > self.sma200[-1]

        if breakout and vol_regime_high and momentum_confirm and trend_filter and not self.position:
            # Inverse volatility scaling
            vol_ratio = self.atr_ma[-1] / max(self.atr[-1], 0.0001)
            size_multiplier = max(self.vol_scale_floor, min(self.vol_scale_ceil, vol_ratio))
            size = self.base_fraction * size_multiplier
            self.buy(size=size)
            self.highest_price = close
            self.entry_price = close
            self.entry_bar = current_bar
        elif self.position:
            # Update running peak
            self.highest_price = max(self.highest_price, close)
            bars_in_position = current_bar - self.entry_bar

            # Regime-adaptive base multiplier
            is_high_vol = self.atr[-1] > self.atr_ma[-1] * self.atr_vol_threshold
            base_mult = self.atr_multiplier_high_vol if is_high_vol else self.atr_multiplier_low_vol

            # Profit-dependent tightening: once profit >= profit_atr_threshold,
            # use a tighter trailing multiplier to lock gains.
            profit_atr = (self.highest_price - self.entry_price) / max(self.atr[-1], 0.0001)
            if profit_atr >= self.profit_atr_threshold:
                base_mult = self.tight_atr_multiplier

            # ATR trailing stop (regime-adaptive only)
            trailing_stop = self.highest_price - self.atr[-1] * base_mult
            stop_hit = close <= trailing_stop

            # Donchian channel exit: price closes below the lower Donchian (exit_period)
            donchian_exit = close < self.exit_lower[-1]

            if stop_hit or donchian_exit:
                self.position.close()
                self.highest_price = None
                self.entry_price = None
                self.entry_bar = None