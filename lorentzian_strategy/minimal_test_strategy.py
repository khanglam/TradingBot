from lumibot.brokers import Alpaca
from lumibot.backtesting import PolygonDataBacktesting
from lumibot.strategies.strategy import Strategy
import datetime as dt
import pytz
import pandas as pd
import numpy as np

class MinimalTestStrategy(Strategy):
    def initialize(self):
        self.sleeptime = "1D"  # Run once per day
        self.set_market("us_equities")
        self.symbols = ["SPY"]  # Just trade SPY
        self.last_trade_date = None
        self.log_message("Strategy initialized")

    def on_trading_iteration(self):
        # Get the current datetime in the market timezone
        current_date = self.get_datetime()
        self.log_message(f"Trading iteration at {current_date}")
        
        # Get portfolio value and cash
        portfolio_value = self.portfolio_value
        cash = self.cash
        symbol = self.symbols[0]
        
        # Log detailed information
        self.log_message(f"Portfolio value: {portfolio_value}")
        self.log_message(f"Cash: {cash}")
        self.log_message(f"Direct cash from get_cash(): {self.get_cash()}")
        
        # Get current position
        position = self.get_position(symbol)
        position_size = 0.1  # Use 10% of cash for each trade
        
        if position is not None:
            quantity = position.quantity
            self.log_message(f"Current position: {position}, quantity: {quantity}")
            
            # Close position every 3 days
            if self.last_trade_date and (current_date.date() - self.last_trade_date).days >= 3:
                self.log_message(f"Closing position for {symbol}, quantity: {abs(quantity)}")
                try:
                    if quantity > 0:  # Long position
                        order = self.create_order(symbol, quantity, "sell")
                    else:  # Short position
                        order = self.create_order(symbol, abs(quantity), "buy")
                    
                    self.log_message(f"Order created: {order}")
                    self.submit_order(order)
                    self.log_message(f"Close order submitted successfully")
                    self.last_trade_date = None
                except Exception as e:
                    self.log_message(f"Error submitting close order: {e}")
        else:
            self.log_message("No current position")
            
            # Get the last price
            price = self.get_last_price(symbol)
            self.log_message(f"Last price for {symbol}: {price}")
            
            # Calculate quantity
            try:
                if price <= 0:
                    self.log_message("Error: Price is zero or negative")
                    quantity = 0
                else:
                    # Use direct cash for calculation
                    direct_cash = self.get_cash()
                    effective_cash = direct_cash if direct_cash is not None else cash
                    self.log_message(f"Using effective cash: {effective_cash}")
                    
                    raw_quantity = (effective_cash * position_size) / price
                    self.log_message(f"Raw quantity calculation: {raw_quantity}")
                    quantity = max(1, int(raw_quantity))  # Ensure at least 1 share
            except Exception as e:
                self.log_message(f"Error calculating quantity: {e}")
                quantity = 1  # Default to 1 share if there's an error
            
            self.log_message(f"Calculated quantity: {quantity}")
            
            # Alternate between buy and sell signals
            day_of_month = current_date.day
            buy_signal = (day_of_month % 2 == 0)  # Buy on even days
            
            if buy_signal:
                self.log_message(f"Opening LONG position for {symbol}, quantity: {quantity}")
                try:
                    order = self.create_order(symbol, quantity, "buy")
                    self.log_message(f"Order created: {order}")
                    self.submit_order(order)
                    self.log_message(f"Buy order submitted successfully")
                    self.last_trade_date = current_date.date()
                except Exception as e:
                    self.log_message(f"Error submitting buy order: {e}")
            else:
                self.log_message(f"Opening SHORT position for {symbol}, quantity: {quantity}")
                try:
                    order = self.create_order(symbol, quantity, "sell")
                    self.log_message(f"Order created: {order}")
                    self.submit_order(order)
                    self.log_message(f"Sell order submitted successfully")
                    self.last_trade_date = current_date.date()
                except Exception as e:
                    self.log_message(f"Error submitting sell order: {e}")


if __name__ == "__main__":
    # Define whether we're backtesting or live trading
    IS_BACKTESTING = True
    
    if IS_BACKTESTING:
        from lumibot.backtesting import PolygonDataBacktesting
        
        # Set up backtesting parameters - use a very short period for testing
        tzinfo = pytz.timezone('America/New_York')
        backtesting_start = tzinfo.localize(dt.datetime(2023, 5, 15))
        backtesting_end = tzinfo.localize(dt.datetime(2023, 5, 25))  # Just 10 days
        
        # Additional backtesting parameters
        timestep = 'day'
        
        # Import API keys from config
        try:
            from config import ALPACA_CONFIG
            # For Polygon, we can use the same API key as Alpaca
            POLYGON_API_KEY = ALPACA_CONFIG.get("API_KEY", "")
            if not POLYGON_API_KEY:
                POLYGON_API_KEY = input("Enter your Polygon API key: ")
        except ImportError:
            POLYGON_API_KEY = input("Enter your Polygon API key: ")
        
        # Set up the data source
        data_source = PolygonDataBacktesting(
            api_key=POLYGON_API_KEY,
            datetime_start=backtesting_start,
            datetime_end=backtesting_end,
            timestep=timestep,
        )
        
        # Set up the strategy
        strategy = MinimalTestStrategy(
            name="MinimalTestStrategy",
            backtesting=data_source,
            analyze_backtest=True,
            parameters={
                "starting_cash": 100000,
            },
            show_progress_bar=True,
        )
        
        # Run the backtest
        strategy.backtest()
    else:
        # Live trading setup (not used in this example)
        pass
