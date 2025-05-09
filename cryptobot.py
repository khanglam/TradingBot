from lumibot.backtesting import CcxtDataBacktesting
from lumibot.credentials import IS_BACKTESTING
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.entities import Order
from lumibot.brokers import Ccxt

from pandas import Timedelta  # For time calculations
import os  # Import os to load environment variables
import pandas as pd
from pandas_ta import atr, rsi  # For technical indicators
import ccxt  # For crypto exchange access

# Load environment variables
CASH_AT_RISK = os.getenv('CASH_AT_RISK', '0.02')  # Default to 2% if not set
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME', 'binance')  # Default to Binance
API_KEY = os.getenv('CRYPTO_API_KEY', '')  # Your crypto exchange API key
API_SECRET = os.getenv('CRYPTO_API_SECRET', '')  # Your crypto exchange API secret

# Crypto configuration
CCXT_CONFIG = {
    "exchange_id": EXCHANGE_NAME,
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}  # For spot trading
}

class CryptoTrader(Strategy):
    def initialize(self, symbol="PEPE/USDT"):
        self.symbol = symbol
        self.sleeptime = "1M"  # Check every minute for crypto's faster markets
        self.set_market("crypto")
        self.positions_data = {}  # To track entry prices and stop/target levels
        
        # Create a direct ccxt instance for additional data if needed
        self.exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True
        })

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

    def get_crypto_volatility(self, length=14):
        """Calculate crypto volatility based on ATR/price ratio"""
        atr_value = self.calculate_atr(length)
        current_price = self.get_last_price(self.symbol)
        
        if atr_value is None or pd.isna(atr_value) or current_price is None:
            return 0.02  # Default to 2% volatility
            
        # Normalize ATR as percentage of price
        return atr_value / current_price
        
    def get_market_trend(self):
        """Determine market trend using RSI and price vs SMA"""
        rsi_value = self.calculate_rsi()
        sma200 = self.get_sma()
        current_price = self.get_last_price(self.symbol)
        
        if sma200 is None or rsi_value is None:
            return "neutral"
            
        # Bullish conditions
        if current_price > sma200 and rsi_value > 50:
            return "bullish"
        # Bearish conditions
        elif current_price < sma200 and rsi_value < 50:
            return "bearish"
        # Neutral conditions
        else:
            return "neutral"

    def on_trading_iteration(self):
        # Get SMA and check if it's available
        sma200 = self.get_sma()
        if sma200 is None:
            self.log_message("SMA calculation failed, skipping iteration")
            return
            
        # Get current position, price, and holdings
        position = self.get_position(self.symbol)
        current_price = self.get_last_price(self.symbol)
        current_holdings = position.quantity if position is not None else 0
        
        # Log current state
        self.log_message(f"Current {self.symbol} price: {current_price}, Holdings: {current_holdings}")
        self.log_message(f"200-day SMA: {sma200}")
        
        # Check for take profit or stop loss if we have a position
        if current_holdings > 0 and self.symbol in self.positions_data:
            position_data = self.positions_data[self.symbol]
            
            # Take profit condition
            if current_price >= position_data["take_profit"]:
                sell_order = self.create_order(self.symbol, current_holdings, Order.OrderSide.SELL)
                self.submit_order(sell_order)
                profit = (current_price - position_data["entry_price"]) * current_holdings
                print(f"\nTAKE PROFIT: Selling {current_holdings} {self.symbol} at ${current_price:.6f}")
                print(f"Entry: ${position_data['entry_price']:.6f}, Profit: ${profit:.6f}")
                del self.positions_data[self.symbol]
                return
                
            # Stop loss condition
            if current_price <= position_data["stop_loss"]:
                sell_order = self.create_order(self.symbol, current_holdings, Order.OrderSide.SELL)
                self.submit_order(sell_order)
                loss = (position_data["entry_price"] - current_price) * current_holdings
                print(f"\nSTOP LOSS: Selling {current_holdings} {self.symbol} at ${current_price:.6f}")
                print(f"Entry: ${position_data['entry_price']:.6f}, Loss: ${loss:.6f}")
                del self.positions_data[self.symbol]
                return
        
        # Get market trend for entry/exit signals
        market_trend = self.get_market_trend()
        self.log_message(f"Current market trend: {market_trend}")
        
        # Calculate volatility for position sizing and stop loss
        volatility = self.get_crypto_volatility()
        atr_value = self.calculate_atr(length=14)
        if atr_value is None or pd.isna(atr_value):
            # Default to 2% of price if ATR calculation fails
            atr_value = current_price * volatility

        # BUY SIGNAL - If 0 holding AND bullish trend
        if current_holdings == 0 and market_trend == "bullish":
            available_cash = self.get_cash()
            # Use a percentage of available cash for crypto (more volatile)
            cash_to_use = available_cash * float(CASH_AT_RISK)
            coins_to_trade = cash_to_use / current_price
            
            # Round to appropriate decimal places for the crypto
            # For small value coins like PEPE, we might need more precision
            coins_to_trade = round(coins_to_trade, 2)  
            
            if coins_to_trade <= 0:
                print(f"\nNot enough cash to buy. Available cash: {available_cash}")
            else:
                # Set stop loss at 2 ATR below entry price
                stop_loss = current_price - (2 * atr_value)
                take_profit = current_price + (3 * atr_value)  # 3:2 reward-to-risk ratio
                
                # Store position data for tracking
                self.positions_data[self.symbol] = {
                    "entry_price": current_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "quantity": coins_to_trade,
                    "entry_date": self.get_datetime()
                }

                # Execute buy order
                buy_order = self.create_order(self.symbol, coins_to_trade, Order.OrderSide.BUY)
                self.submit_order(buy_order)
                print(f"\nBUY ORDER: {coins_to_trade} {self.symbol} at ${current_price:.6f}")
                print(f"Stop Loss: ${stop_loss:.6f}, Take Profit: ${take_profit:.6f}")
                print(f"Risk: ${(current_price - stop_loss) * coins_to_trade:.6f}, Reward: ${(take_profit - current_price) * coins_to_trade:.6f}")

        # SELL SIGNAL - If we have a position AND bearish trend
        elif current_holdings > 0 and market_trend == "bearish":
            sell_order = self.create_order(self.symbol, current_holdings, Order.OrderSide.SELL)
            self.submit_order(sell_order)
            
            # Calculate profit/loss if we have the entry data
            if self.symbol in self.positions_data:
                entry_price = self.positions_data[self.symbol]["entry_price"]
                profit = (current_price - entry_price) * current_holdings
                print(f"\nSELL SIGNAL: Selling {current_holdings} {self.symbol} at ${current_price:.6f}")
                print(f"Entry: ${entry_price:.6f}, P/L: ${profit:.6f}")
                del self.positions_data[self.symbol]
            else:
                print(f"\nSELL SIGNAL: Selling {current_holdings} {self.symbol} at ${current_price:.6f}")

# Initialize the CCXT broker
broker = Ccxt(CCXT_CONFIG)

if __name__ == "__main__":
    IS_BACKTESTING = False
    if IS_BACKTESTING:
        symbol = "PEPE/USDT"
        result = CryptoTrader.backtest(
            CcxtDataBacktesting,
            symbol=symbol,
            benchmark_asset="BTC/USDT",  # Use BTC as benchmark
            quote_asset="USDT",
            exchange_name=EXCHANGE_NAME,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
    else:
        strategy = CryptoTrader(broker=broker, symbol="PEPE/USDT")
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