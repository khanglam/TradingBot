from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import PolygonDataBacktesting
import datetime as dt
import pytz

class SuperSimpleStrategy(Strategy):
    def initialize(self):
        self.sleeptime = "1D"  # Run once per day
        self.set_market("us_equities")
        self.symbols = ["SPY"]  # Just trade SPY
        self.log_message("Strategy initialized")

    def on_trading_iteration(self):
        # Get the current datetime in the market timezone
        current_date = self.get_datetime()
        self.log_message(f"Trading iteration at {current_date}")
        
        # Get symbol
        symbol = self.symbols[0]
        
        # Get portfolio value and cash
        portfolio_value = self.portfolio_value
        cash = self.cash
        direct_cash = self.get_cash()
        
        # Log detailed information
        self.log_message(f"DETAILED DEBUG - Portfolio value: {portfolio_value}, Cash: {cash}, Direct cash: {direct_cash}")
        self.log_message(f"Broker type: {type(self.broker)}")
        self.log_message(f"Is backtesting: {self.is_backtesting}")
        
        # Get position
        position = self.get_position(symbol)
        self.log_message(f"Current position: {position}")
        
        # Get orders
        orders = self.get_orders()
        self.log_message(f"Current orders: {orders}")
        
        # ALWAYS try to buy 1 share on every iteration
        quantity = 1
        price = self.get_last_price(symbol)
        self.log_message(f"Current price for {symbol}: {price}")
        
        # Only buy if we don't have a position
        if position is None:
            self.log_message(f"Attempting to buy {quantity} shares of {symbol}")
            try:
                # Create the order
                order = self.create_order(symbol, quantity, "buy")
                self.log_message(f"Order created: {order}")
                
                # Submit the order
                result = self.submit_order(order)
                self.log_message(f"Order submission result: {result}")
                
                # Check orders after submission
                orders_after = self.get_orders()
                self.log_message(f"Orders after submission: {orders_after}")
            except Exception as e:
                self.log_message(f"Error creating/submitting order: {e}")
                import traceback
                self.log_message(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Define whether we're backtesting or live trading
    IS_BACKTESTING = True
    
    if IS_BACKTESTING:
        # Set up backtesting parameters - use a very short period for testing
        tzinfo = pytz.timezone('America/New_York')
        backtesting_start = tzinfo.localize(dt.datetime(2023, 5, 15))
        backtesting_end = tzinfo.localize(dt.datetime(2023, 5, 20))  # Very short period for testing
        
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
        strategy = SuperSimpleStrategy(
            name="SuperSimpleStrategy",
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
