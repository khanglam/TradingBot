from math import log
from typing import List

import pandas as pd
import numpy as np  # Needed for fast numerical work

from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order, TradingFee
from lumibot.backtesting import PolygonDataBacktesting  # Polygon works for both stocks & options / minute data
from lumibot.credentials import IS_BACKTESTING

"""
Strategy Description
--------------------
This is an enhanced version of the “Lorentzian Classification” idea.  It still
compares today’s indicator mix (RSI, Wave-Trend, CCI) with the *k* most similar
historical days, but the way we build the “training set” has been corrected so
that every historical row is labelled **by what happened NEXT** (up-move or
down-move on the following day).  That immediately produces a lot more buy /
sell signals.

Fix Applied (2024-06-24)
------------------------
Users running recent versions of pandas were getting:
    AttributeError: 'Series' object has no attribute 'mad'
The `mad()` helper vanished in pandas 2.0.  We now calculate the mean absolute
deviation manually, so the strategy runs on **all** pandas versions again.

This code was refined based on the user prompt: "Getting issues with this
strategy. Not able to generate any trades".
"""

# -------------------------------  Indicator helpers  ----------------------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Classic RSI calculation (identical to TradingView)."""
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def _rolling_mad(arr: np.ndarray) -> float:
    """Helper that returns mean-absolute-deviation for a numpy array."""
    mean_val = np.nanmean(arr)
    return np.nanmean(np.abs(arr - mean_val))

def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Commodity Channel Index – TradingView formula.

    Re-implemented without the deprecated pandas .mad() to guarantee
    compatibility with pandas ≥ 2.0.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(length).mean()

    # Calculate MAD (mean absolute deviation) *manually* because .mad() is gone.
    mad = tp.rolling(length).apply(_rolling_mad, raw=True)

    return (tp - sma) / (0.015 * mad)

def wavetrend(df: pd.DataFrame, channel_length: int = 10, average_length: int = 11) -> pd.Series:
    """Light-weight Wave-Trend approximation using two EMAs."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa = hlc3.ewm(span=channel_length, adjust=False).mean()
    de = np.abs(hlc3 - esa).ewm(span=channel_length, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * de)
    return ci.ewm(span=average_length, adjust=False).mean()

# -------------------------------  Strategy class  -------------------------------------

class LorentzianClassificationStrategy(Strategy):
    parameters = {
        "symbols": ["TSLA"],      # You can override with any list of tickers
        "neighbors": 8,          # k in k-Nearest-Neighbours
        "history_window": 300,   # how many past days to feed into the model
        "rsi_length": 14,
        "wt_channel": 10,
        "wt_average": 11,
        "cci_length": 20,
    }

    # ---------------------------  Initialise (runs once)  ---------------------------
    def initialize(self):
        # Run the bot once per trading day – ideal for end-of-day swing signals
        self.sleeptime = "1D"

        # Respect market setting from environment (e.g. 24/7 for crypto) but
        # fall back to NYSE if user didn’t override it.
        self.set_market("NYSE")
        
        p = self.parameters
        self.symbols: List[str] = p.get("symbols")
        self.k = int(p.get("neighbors"))
        self.window = int(p.get("history_window"))
        self.rsi_len = int(p.get("rsi_length"))
        self.wt_channel = int(p.get("wt_channel"))
        self.wt_average = int(p.get("wt_average"))
        self.cci_len = int(p.get("cci_length"))

        # Keep track of previous signal so we only draw a marker when it flips
        if not hasattr(self.vars, "prev_signal"):
            self.vars.prev_signal = {sym: 0 for sym in self.symbols}

    # ---------------------------  Main daily loop  ----------------------------------
    def on_trading_iteration(self):
        for symbol in self.symbols:
            asset = Asset(symbol, Asset.AssetType.STOCK)

            # 1) Fetch price history (need window + 1 days to know “tomorrow”)
            bars = self.get_historical_prices(asset, self.window + 1, "day")
            if bars is None or len(bars.df) < self.window + 1:
                self.log_message(f"Not enough data for {symbol}, waiting…", color="yellow")
                continue
            df = bars.df.copy()

            # 2) Compute indicators for the entire window
            df["RSI"] = rsi(df["close"], self.rsi_len)
            df["WT"] = wavetrend(df, self.wt_channel, self.wt_average)
            df["CCI"] = cci(df, self.cci_len)

            # 3) Build (features, label) set – label is *tomorrow’s* move
            df["label"] = 0  # default
            df.loc[df["close"].shift(-1) > df["close"], "label"] = 1
            df.loc[df["close"].shift(-1) < df["close"], "label"] = -1
            dataset = df.dropna(subset=["RSI", "WT", "CCI", "label"]).iloc[:-1]
            if dataset.empty:
                self.log_message(f"Indicators warming up for {symbol}", color="yellow")
                continue

            # 4) Today’s feature vector
            today = df.iloc[-1]
            current_features = [today["RSI"], today["WT"], today["CCI"]]

            # 5) k-NN using Lorentzian distance
            distances = []
            for _, row in dataset.iterrows():
                d = sum(log(1 + abs(a - b)) for a, b in zip(current_features, [row["RSI"], row["WT"], row["CCI"]]))
                distances.append((d, row["label"]))
            distances.sort(key=lambda x: x[0])
            nearest = distances[: self.k]

            prediction_score = sum(lbl for _, lbl in nearest)
            if prediction_score > 0:
                prediction = 1
            elif prediction_score < 0:
                prediction = -1
            else:
                # Tie – fall back to simple one-day momentum
                prediction = 1 if df["close"].iloc[-1] > df["close"].iloc[-2] else -1

            # 6) Charting aids (price line + marker when signal flips)
            self.add_line(symbol, today["close"], color="black")  # price line
            if prediction != self.vars.prev_signal[symbol]:
                marker_color = "green" if prediction == 1 else "red"
                marker_symbol = "arrow-up" if prediction == 1 else "arrow-down"
                self.add_marker(f"{symbol} signal", today["close"], color=marker_color, symbol=marker_symbol, size=10)
                self.vars.prev_signal[symbol] = prediction

            # 7) Trading logic
            position = self.get_position(asset)
            price = today["close"]

            if prediction == 1 and (position is None or position.quantity == 0):
                cash = self.get_cash()
                qty = int((cash * 0.99) // price)  # invest ~99 % of cash
                if qty > 0:
                    order = self.create_order(asset, qty, Order.OrderSide.BUY)
                    self.submit_order(order)
                    self.log_message(f"BUY {qty} {symbol} @ {price:.2f}", color="green")
                else:
                    self.log_message("Not enough cash to buy even one share", color="yellow")

            elif prediction == -1 and position is not None and position.quantity > 0:
                order = self.create_order(asset, position.quantity, Order.OrderSide.SELL)
                self.submit_order(order)
                self.log_message(f"SELL {position.quantity} {symbol} @ {price:.2f}", color="red")

# -------------------------------  Runner  ---------------------------------------------
if __name__ == "__main__":
    params = LorentzianClassificationStrategy.parameters  # can be overridden in UI / CLI

    if IS_BACKTESTING:
        trading_fee = TradingFee(percent_fee=0.001)
        LorentzianClassificationStrategy.backtest(
            datasource_class=PolygonDataBacktesting,  # Polygon gives us robust daily/minute data
            benchmark_asset=Asset("TSLA", Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=params,
        )
    else:
        trader = Trader()
        strategy = LorentzianClassificationStrategy(
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=params,
        )
        trader.add_strategy(strategy)
        trader.run_all()