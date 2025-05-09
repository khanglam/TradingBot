from lumibot.backtesting import PolygonDataBacktesting
from lumibot.credentials import IS_BACKTESTING, ALPACA_CONFIG
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.entities import Order
from lumibot.brokers import Alpaca

from alpaca_trade_api import REST  # Alpaca's REST API client
from pandas import Timedelta  # For time calculations
from finbert_utils import estimate_sentiment  # For sentiment analysis
import os  # Import os to load environment variables
import pandas as pd
from pandas_ta import atr, rsi  # Add to imports

# Load environment variables
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')
CASH_AT_RISK = os.getenv('CASH_AT_RISK')
PROBABILITY_THRESHOLD = float(os.getenv('PROBABILITY_THRESHOLD'))

class MLTrader(Strategy):
    def initialize(self, symbol="TSLA"):
        self.symbol = symbol
        self.sleeptime = "5M"
        self.set_market("NYSE")
        self.positions_data = {} # To track entry prices and stop/target levels
        self.api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_CONFIG['API_KEY'], secret_key=ALPACA_CONFIG['API_SECRET'])

    def get_sma(self):
        # Request 210 days to ensure at least 200 valid days for SMA calculation
        bars = self.get_historical_prices(self.symbol, 210, 'day')
        if bars is None:
            self.log_message('Historical data unavailable for ' + self.symbol)
            return

        # Convert the historical data into a DataFrame
        df = bars.df
        if df.shape[0] < 200:
            self.log_message('Not enough historical data to compute 200-day SMA.')
            return

        # Calculate the 200-day simple moving average (SMA) using closing prices
        sma200 = df['close'].tail(200).mean()
        self.log_message(f'Calculated 200-day SMA: {sma200:.2f}')
        return sma200

    def calculate_atr(self, length=14):
        bars = self.get_historical_prices(self.symbol, length+1, "day")
        if bars is None:
            return None
        # Access the DataFrame through the .df attribute
        df = bars.df
        if df.empty:
            return None
        try:
            return atr(df['high'], df['low'], df['close'], length=length).iloc[-1]
        except Exception as e:
            self.log_message(f"Error calculating ATR: {str(e)}")
            return None  # Return None and handle the default in on_trading_iteration

    def calculate_rsi(self, length=14):
        bars = self.get_historical_prices(self.symbol, length+1, "day") 
        if bars is None:
            return None
        # Access the DataFrame through the .df attribute
        df = bars.df
        if df.empty:
            return None
        try:
            return rsi(df['close'], length=length).iloc[-1]
        except Exception as e:
            self.log_message(f"Error calculating RSI: {str(e)}")
            return 50  # Default to neutral RSI

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()  # Get date range
        # Fetch news articles for the stock and cache
        if not hasattr(self, "last_news_date") or self.last_news_date != today:
            news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
            self.news_headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
            self.last_news_date = today
        # Analyze sentiment of the news articles
        return estimate_sentiment(self.news_headlines)

    def on_trading_iteration(self):
        # Get SMA and check if it's available
        sma200 = self.get_sma()
        if sma200 is None:
            return
            
        # Get current position, price, and holdings
        position = self.get_position(self.symbol)
        current_price = self.get_last_price(self.symbol)
        current_holdings = position.quantity if position is not None else 0
        
        # Check for take profit or stop loss if we have a position
        if current_holdings > 0 and self.symbol in self.positions_data:
            position_data = self.positions_data[self.symbol]
            
            # Take profit condition
            if current_price >= position_data["take_profit"]:
                sell_order = self.create_order(self.symbol, current_holdings, Order.OrderSide.SELL)
                self.submit_order(sell_order)
                profit = (current_price - position_data["entry_price"]) * current_holdings
                print(f"\nTAKE PROFIT: Selling {current_holdings} shares at ${current_price:.2f}")
                print(f"Entry: ${position_data['entry_price']:.2f}, Profit: ${profit:.2f}")
                del self.positions_data[self.symbol]
                return
                
            # Stop loss condition
            if current_price <= position_data["stop_loss"]:
                sell_order = self.create_order(self.symbol, current_holdings, Order.OrderSide.SELL)
                self.submit_order(sell_order)
                loss = (position_data["entry_price"] - current_price) * current_holdings
                print(f"\nSTOP LOSS: Selling {current_holdings} shares at ${current_price:.2f}")
                print(f"Entry: ${position_data['entry_price']:.2f}, Loss: ${loss:.2f}")
                del self.positions_data[self.symbol]
                return
        
        # Get sentiment for entry/exit signals
        probability, sentiment = self.get_sentiment()
        
        # Calculate ATR for position sizing and stop loss
        atr_value = self.calculate_atr(length=14)
        if atr_value is None or pd.isna(atr_value):
            # Default to 2% of price if ATR calculation fails
            atr_value = current_price * 0.02

        # BUY SIGNAL - If 0 holding AND current price is above SMA AND sentiment is positive
        if current_holdings == 0 and current_price > sma200:# and sentiment == "positive" and probability > PROBABILITY_THRESHOLD:
            available_cash = self.get_cash()
            shares_to_trade = int(available_cash // current_price)
            if shares_to_trade <= 0:
                print("\nNot enough cash to buy. Available cash: " + str(available_cash))
            else:
                # Set stop loss at 2 ATR below entry price
                stop_loss = current_price - (2 * atr_value)
                take_profit = current_price + (3 * atr_value)  # 3:2 reward-to-risk ratio
                
                # Store position data for tracking
                self.positions_data[self.symbol] = {
                    "entry_price": current_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "quantity": shares_to_trade,
                    "entry_date": self.get_datetime()
                }

                # Execute buy order
                buy_order = self.create_order(self.symbol, shares_to_trade, Order.OrderSide.BUY)
                self.submit_order(buy_order)
                print(f"\nBUY ORDER: {shares_to_trade} shares of {self.symbol} at ${current_price:.2f}")
                print(f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
                print(f"Risk: ${(current_price - stop_loss) * shares_to_trade:.2f}, Reward: ${(take_profit - current_price) * shares_to_trade:.2f}")

        # SELL SIGNAL - If we have a position AND current price is below SMA AND sentiment is negative
        elif current_holdings > 0 and current_price < sma200:# and sentiment == "negative" and probability > PROBABILITY_THRESHOLD:
            sell_order = self.create_order(self.symbol, current_holdings, Order.OrderSide.SELL)
            self.submit_order(sell_order)
            
            # Calculate profit/loss if we have the entry data
            if self.symbol in self.positions_data:
                entry_price = self.positions_data["entry_price"]
                profit = (current_price - entry_price) * current_holdings
                print(f"\nSELL SIGNAL: Selling {current_holdings} shares at ${current_price:.2f}")
                print(f"Entry: ${entry_price:.2f}, P/L: ${profit:.2f}")
                del self.positions_data[self.symbol]
            else:
                print(f"\nSELL SIGNAL: Selling {current_holdings} shares at ${current_price:.2f}")

broker = Alpaca(ALPACA_CONFIG)

if __name__ == "__main__":
    IS_BACKTESTING=False
    if IS_BACKTESTING:
        symbol = "TSLA"
        result = MLTrader.backtest(
            PolygonDataBacktesting,
            symbol=symbol,
            benchmark_asset=symbol,
        )
    else:
        strategy = MLTrader(broker=broker)
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()


        # symbol = "TSLA"
        # timestep = 'day'
        # auto_adjust = True
        # warm_up_trading_days = 0
        # refresh_cache = False

        # results, strategy = MLTrader.run_backtest(
        #     datasource_class=PolygonDataBacktesting,
        #     cash_at_risk=float(CASH_AT_RISK),
        #     minutes_before_closing=0,
        #     symbol=symbol,
        #     benchmark_asset='SPY',
        #     analyze_backtest=True,
        #     show_progress_bar=True,

        #     # AlpacaBacktesting kwargs
        #     timestep=timestep,
        #     market='NYSE',
        #     config=ALPACA_CONFIG,
        #     refresh_cache=refresh_cache,
        #     warm_up_trading_days=warm_up_trading_days,
        #     # auto_adjust=auto_adjust,
        # )