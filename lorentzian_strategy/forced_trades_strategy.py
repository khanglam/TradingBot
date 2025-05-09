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

class ForcedTradesStrategy(Strategy):
    """A simple strategy that forces trades on specific days for testing purposes."""
    
    parameters = {
        "symbol": "SPY",
        "position_size": 1.0,  # Percentage of portfolio to allocate per trade (0.0-1.0)
    }
    
    def initialize(self):
        self.sleeptime = "1D"  # Run the strategy once per day
        self.last_trade_date = None
    
    def on_trading_iteration(self):
        # Get current datetime and symbol
        current_dt = self.get_datetime()
        symbol = self.parameters["symbol"]
        
        # Log current datetime and portfolio information
        print(f"STRATEGY LOG: Current datetime: {current_dt}")
        print(f"STRATEGY LOG: Portfolio value: {self.portfolio_value}, Cash: {self.cash}")
        
        # Use both print and log_message for maximum visibility
        self.log_message(f"Current datetime: {current_dt}")
        self.log_message(f"Portfolio value: {self.portfolio_value}, Cash: {self.cash}")
        
        # Get current position
        current_position = self.get_position(symbol)
        self.log_message(f"Current position: {current_position}")
        
        # Force a buy signal on EVERY day for testing
        day_of_month = current_dt.day
        force_buy = True  # Force buy on every day
        force_sell = (day_of_month % 7 == 0)  # Still sell on days divisible by 7
        
        print(f"STRATEGY LOG: FORCING BUY SIGNAL FOR TESTING on {current_dt}")
        
        # Log the signal decision
        self.log_message(f"Day of month: {day_of_month}, Force buy: {force_buy}, Force sell: {force_sell}")
        
        # Get the current price
        price = self.get_last_price(symbol)
        self.log_message(f"Current price for {symbol}: {price}")
        
        # Trading logic
        if current_position is None:
            # No current position, check for new signals
            if force_buy:
                # Calculate quantity based on position size
                position_size = self.parameters["position_size"]
                
                # Debug cash value
                self.log_message(f"Cash before calculation: {self.cash}, Position size: {position_size}")
                
                # Calculate quantity with extra checks
                try:
                    if price <= 0:
                        self.log_message("Error: Price is zero or negative")
                        quantity = 0
                    else:
                        raw_quantity = (self.cash * position_size) / price
                        self.log_message(f"Raw quantity calculation: {raw_quantity}")
                        quantity = int(raw_quantity)
                except Exception as e:
                    self.log_message(f"Error calculating quantity: {e}")
                    quantity = 0
                
                self.log_message(f"Buy signal - Price: {price}, Cash available: {self.cash}, Calculated quantity: {quantity}")
                
                # Try with a hardcoded quantity instead of calculated quantity
                hardcoded_quantity = 10  # Just try with 10 shares
                self.log_message(f"Opening LONG position for {symbol}, HARDCODED quantity: {hardcoded_quantity}")
                print(f"STRATEGY LOG: Attempting to buy {hardcoded_quantity} shares of {symbol} at {price}")
                
                try:
                    order = self.create_order(symbol, hardcoded_quantity, "buy")
                    self.submit_order(order)
                    print(f"STRATEGY LOG: Order submitted successfully: {order}")
                    self.last_trade_date = current_dt.date()
                except Exception as e:
                    print(f"STRATEGY LOG: ERROR submitting order: {e}")
                    self.log_message(f"Error submitting order: {e}")
        
        else:
            # We have a position, check for exit signals
            quantity = current_position.quantity
            
            # Check if position is long (quantity > 0) or short (quantity < 0)
            is_long_position = quantity > 0
            is_short_position = quantity < 0
            
            # Log position details
            self.log_message(f"Position details - quantity: {quantity}, is_long: {is_long_position}, is_short: {is_short_position}")
            
            # Only sell if we've held the position for at least one day
            if self.last_trade_date is not None and current_dt.date() > self.last_trade_date:
                if is_long_position and force_sell:
                    self.log_message(f"Closing LONG position for {symbol}, quantity: {quantity}")
                    order = self.create_order(symbol, quantity, "sell")
                    self.submit_order(order)
                    self.last_trade_date = None


if __name__ == "__main__":
    # Set up backtesting parameters
    tzinfo = pytz.timezone('America/New_York')
    backtesting_start = tzinfo.localize(dt.datetime(2023, 5, 15))
    backtesting_end = tzinfo.localize(dt.datetime(2023, 6, 15))  # Shorter period for testing
    
    # Additional backtesting parameters
    timestep = 'day'
    auto_adjust = True
    warm_up_trading_days = 0  # No warm-up period needed for this test
    refresh_cache = True  # Force refresh of data cache
    
    # Run the backtest
    results, strategy = ForcedTradesStrategy.run_backtest(
        datasource_class=PolygonDataBacktesting,  # Switch to Polygon
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        minutes_before_closing=0,
        benchmark_asset='SPY',
        analyze_backtest=True,
        parameters={
            "symbol": "SPY",
            "position_size": 0.9,  # Use 90% of cash for each trade
        },
        show_progress_bar=True,
        
        # PolygonDataBacktesting kwargs
        timestep=timestep,
        market='NYSE',
        config=ALPACA_CONFIG,
        refresh_cache=refresh_cache,
    )
    
    # Print the results
    print(results)
