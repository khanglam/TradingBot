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

class BasicTestStrategy(Strategy):
    """A minimal strategy that attempts to create trades on every iteration."""
    
    parameters = {
        "symbol": "SPY",
        "position_size": 0.1,  # Percentage of portfolio to allocate per trade (0.0-1.0)
    }
    
    def initialize(self):
        self.sleeptime = "1D"  # Run the strategy once per day
        self.last_trade_date = None
    
    def on_trading_iteration(self):
        # Get current datetime and symbol
        current_dt = self.get_datetime()
        symbol = self.parameters["symbol"]
        
        # Log current datetime and portfolio information
        self.log_message(f"Current datetime: {current_dt}")
        self.log_message(f"Portfolio value: {self.portfolio_value}")
        self.log_message(f"Cash: {self.cash}")
        
        # Get current position
        position = self.get_position(symbol)
        self.log_message(f"Current position: {position}")
        
        # Get the last price
        price = self.get_last_price(symbol)
        self.log_message(f"Current price for {symbol}: {price}")
        
        # Always try to create a trade on every iteration
        if position is None:
            # No position, create a buy order
            self.log_message("No current position, creating buy order")
            
            # Calculate quantity based on position size
            position_size = self.parameters["position_size"]
            cash = self.cash
            
            # Calculate quantity
            try:
                if price <= 0:
                    self.log_message("Error: Price is zero or negative")
                    quantity = 0
                else:
                    raw_quantity = (cash * position_size) / price
                    self.log_message(f"Raw quantity calculation: {raw_quantity}")
                    quantity = int(raw_quantity)
            except Exception as e:
                self.log_message(f"Error calculating quantity: {e}")
                quantity = 0
            
            # Force a minimum quantity for testing
            if quantity == 0 and cash > 0:
                self.log_message("Forcing minimum quantity of 1 for testing")
                quantity = 1
            
            self.log_message(f"Creating buy order for {quantity} shares of {symbol}")
            
            # Create and submit the order
            try:
                order = self.create_order(symbol, quantity, "buy")
                self.log_message(f"Order created: {order}")
                self.submit_order(order)
                self.log_message("Order submitted successfully")
                self.last_trade_date = current_dt.date()
            except Exception as e:
                self.log_message(f"Error submitting order: {e}")
        
        else:
            # Already have a position, check if we should close it
            quantity = position.quantity
            self.log_message(f"Current position quantity: {quantity}")
            
            # Close position if it's been held for at least 2 days
            if self.last_trade_date and (current_dt.date() - self.last_trade_date).days >= 2:
                self.log_message(f"Closing position for {symbol}, quantity: {quantity}")
                
                try:
                    if quantity > 0:  # Long position
                        order = self.create_order(symbol, quantity, "sell")
                    else:  # Short position
                        order = self.create_order(symbol, abs(quantity), "buy")
                        
                    self.log_message(f"Close order created: {order}")
                    self.submit_order(order)
                    self.log_message("Close order submitted successfully")
                    self.last_trade_date = None
                except Exception as e:
                    self.log_message(f"Error submitting close order: {e}")


if __name__ == "__main__":
    # Set up backtesting parameters
    tzinfo = pytz.timezone('America/New_York')
    backtesting_start = tzinfo.localize(dt.datetime(2023, 5, 15))
    backtesting_end = tzinfo.localize(dt.datetime(2023, 5, 25))  # Short period for testing
    
    # Additional backtesting parameters
    timestep = 'day'
    auto_adjust = True
    warm_up_trading_days = 0  # No warm-up period needed for this test
    refresh_cache = True  # Force refresh of data cache
    
    # Run the backtest
    results, strategy = BasicTestStrategy.run_backtest(
        datasource_class=PolygonDataBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        minutes_before_closing=0,
        benchmark_asset='SPY',
        analyze_backtest=True,
        parameters={
            "symbol": "SPY",
            "position_size": 0.1,  # Use 10% of cash for each trade
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
