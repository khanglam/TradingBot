import datetime as dt
import pandas as pd
import numpy as np
import pytz
import sys

# Import Lumibot classes
from lumibot.backtesting import PolygonDataBacktesting
from lumibot.strategies.strategy import Strategy

# Import Alpaca API config
from config import ALPACA_CONFIG

class UltraSimpleStrategy(Strategy):
    """An ultra simple strategy that buys on even days and sells on odd days."""
    
    def initialize(self):
        self.sleeptime = "1D"  # Run the strategy once per day
        self.symbol = "SPY"  # Always trade SPY
        print("Strategy initialized")
    
    def on_trading_iteration(self):
        # Get current datetime
        current_dt = self.get_datetime()
        print(f"Trading iteration at {current_dt}")
        
        # Get current position
        position = self.get_position(self.symbol)
        print(f"Current position: {position}")
        
        # Get the day of the month to determine buy/sell
        day_of_month = current_dt.day
        is_even_day = day_of_month % 2 == 0
        
        # Buy on even days, sell on odd days
        if is_even_day and position is None:
            # Even day and no position - BUY
            print(f"Even day ({day_of_month}), creating buy order")
            
            # Always buy 1 share
            quantity = 1
            
            try:
                order = self.create_order(self.symbol, quantity, "buy")
                print(f"Order created: {order}")
                self.submit_order(order)
                print("Buy order submitted successfully")
            except Exception as e:
                print(f"Error submitting buy order: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
        
        elif not is_even_day and position is not None:
            # Odd day and have position - SELL
            quantity = position.quantity
            print(f"Odd day ({day_of_month}), closing position, quantity: {quantity}")
            
            try:
                order = self.create_order(self.symbol, quantity, "sell")
                print(f"Order created: {order}")
                self.submit_order(order)
                print("Sell order submitted successfully")
            except Exception as e:
                print(f"Error submitting sell order: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    # Set up backtesting parameters
    tzinfo = pytz.timezone('America/New_York')
    backtesting_start = tzinfo.localize(dt.datetime(2023, 5, 15))
    backtesting_end = tzinfo.localize(dt.datetime(2023, 5, 25))  # Short period for testing
    
    # Run the backtest
    results, strategy = UltraSimpleStrategy.run_backtest(
        datasource_class=PolygonDataBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        minutes_before_closing=0,
        benchmark_asset='SPY',
        parameters={},
        
        # PolygonDataBacktesting kwargs
        config=ALPACA_CONFIG,  # Use config instead of api_key
        timestep="day",
    )
    
    # Print the results
    print(results)
