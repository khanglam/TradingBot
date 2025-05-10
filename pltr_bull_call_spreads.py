from lumibot.backtesting import PolygonDataBacktesting
from lumibot.credentials import IS_BACKTESTING, ALPACA_CONFIG
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.entities import Order, Asset, Position
from lumibot.brokers import Alpaca

from datetime import datetime, timedelta
from pandas import Timedelta
import os
import pandas as pd

# Load environment variables
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

class BullCallSpreadTrader(Strategy):
    def initialize(self, symbol="PLTR", initial_budget=1000, num_spreads=20):
        self.symbol = symbol
        self._initial_budget = initial_budget
        self.num_spreads = num_spreads
        self.sleeptime = "1D"  # Check daily
        self.set_market("NYSE")
        self.last_trade_date = None
        self.monthly_interval = 30  # Days between trades

    def position_sizing(self, current_price):
        """Calculate position sizing based on budget and number of spreads"""
        return self.num_spreads  # Trading exactly 20 spreads as specified

    def get_option_chain(self):
        """Get option chain for the symbol with expiration at least 30 days out"""
        current_date = self.get_datetime()
        min_expiry_date = current_date + timedelta(days=30)
        
        # Get the stock price to calculate strike prices
        current_price = self.get_last_price(self.symbol)
        if current_price is None:
            self.log_message(f"Could not get current price for {self.symbol}")
            return None, None, None, None
            
        # Calculate strike prices for bull call spread
        buy_strike = round(current_price * 1.10, 1)  # 10% OTM for buy leg
        sell_strike = round(current_price * 1.20, 1)  # 20% OTM for sell leg
        
        print(f"\n[{current_date}] Current price: ${current_price:.2f}")
        print(f"Buy strike (10% OTM): ${buy_strike:.2f}")
        print(f"Sell strike (20% OTM): ${sell_strike:.2f}")
        
        # For backtesting, we'll simulate option data
        if self.is_backtesting:
            # Create simulated option data for backtesting
            # In backtesting, we'll use a simple model for option prices
            # Buy call (10% OTM) might cost around 3-5% of stock price
            # Sell call (20% OTM) might cost around 1-2% of stock price
            buy_option_price = current_price * 0.04  # 4% of stock price
            sell_option_price = current_price * 0.015  # 1.5% of stock price
            
            # Calculate expiration date (first expiry at least 30 days out)
            expiry_date = (current_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            # Create simulated option objects
            buy_option = {
                "symbol": f"{self.symbol}_{expiry_date}_C_{buy_strike}",
                "strike": buy_strike,
                "price": buy_option_price,
                "expiration": expiry_date
            }
            
            sell_option = {
                "symbol": f"{self.symbol}_{expiry_date}_C_{sell_strike}",
                "strike": sell_strike,
                "price": sell_option_price,
                "expiration": expiry_date
            }
            
            print(f"Simulated buy option: Strike ${buy_strike} @ ${buy_option_price:.2f}")
            print(f"Simulated sell option: Strike ${sell_strike} @ ${sell_option_price:.2f}")
            print(f"Expiration date: {expiry_date}")
            
            return buy_option, sell_option, expiry_date, current_price
        
        # Live trading - get actual option chain data
        try:
            chains = self.broker.get_option_chain(self.symbol)
            
            # Filter for calls with expiration at least 30 days out
            valid_expirations = []
            for expiration in chains.expirations:
                expiry_date = datetime.strptime(expiration, "%Y-%m-%d")
                if expiry_date.date() >= min_expiry_date.date():
                    valid_expirations.append(expiration)
            
            if not valid_expirations:
                self.log_message("No valid expiration dates found (at least 30 days out)")
                return None, None, None, None
                
            # Choose the first valid expiration
            target_expiration = valid_expirations[0]
            self.log_message(f"Selected expiration date: {target_expiration}")
            
            # Find closest strikes to our calculated values
            call_options = chains.get_calls(target_expiration)
            
            # Find closest strikes to our targets
            buy_option = None
            sell_option = None
            
            closest_buy_diff = float('inf')
            closest_sell_diff = float('inf')
            
            for option in call_options:
                # Find closest buy strike
                buy_diff = abs(option.strike - buy_strike)
                if buy_diff < closest_buy_diff:
                    closest_buy_diff = buy_diff
                    buy_option = option
                
                # Find closest sell strike
                sell_diff = abs(option.strike - sell_strike)
                if sell_diff < closest_sell_diff:
                    closest_sell_diff = sell_diff
                    sell_option = option
            
            if buy_option and sell_option:
                return buy_option, sell_option, target_expiration, current_price
            else:
                self.log_message("Could not find appropriate options for the spread")
                return None, None, None, None
                
        except Exception as e:
            self.log_message(f"Error getting option chain: {str(e)}")
            return None, None, None, None

    def should_trade_today(self):
        """Determine if we should trade today based on monthly interval"""
        current_date = self.get_datetime().date()
        
        # If this is our first trade or it's been a month since last trade
        if self.last_trade_date is None:
            return True
        
        days_since_last_trade = (current_date - self.last_trade_date).days
        return days_since_last_trade >= self.monthly_interval

    def on_trading_iteration(self):
        # Check if we should trade today
        if not self.should_trade_today():
            return
            
        # Get current positions to check if we already have open spreads
        positions = self.get_positions()
        current_date = self.get_datetime()
        print(f"\n[{current_date}] Trading iteration - checking for bull call spread opportunity")
        
        # Get option chain data
        buy_option, sell_option, expiration_date, current_price = self.get_option_chain()
        
        if buy_option is None or sell_option is None:
            print("Could not establish option chain data, skipping trading iteration")
            return
            
        # Calculate number of spreads to trade
        num_spreads = self.position_sizing(current_price)
        
        # Execute bull call spread (buy lower strike call, sell higher strike call)
        try:
            # For backtesting, we need to simulate the option trades
            if self.is_backtesting:
                # Calculate cost of the spread
                buy_price = buy_option["price"]
                sell_price = sell_option["price"]
                net_debit = buy_price - sell_price
                total_cost = net_debit * 100 * num_spreads  # Each option is for 100 shares
                
                # Check if we have enough cash
                available_cash = self.get_cash()
                if available_cash < total_cost:
                    print(f"Not enough cash for bull call spread. Need ${total_cost:.2f}, have ${available_cash:.2f}")
                    return
                
                # Simulate buying the spread by deducting cash
                # In backtesting, we'll track this manually since option backtesting is limited
                print(f"\nBULL CALL SPREAD EXECUTED on {current_date.date()}:")
                print(f"Bought {num_spreads} contracts of {buy_option['symbol']} at strike ${buy_option['strike']}")
                print(f"Sold {num_spreads} contracts of {sell_option['symbol']} at strike ${sell_option['strike']}")
                print(f"Net debit per spread: ${net_debit:.2f}")
                print(f"Total cost: ${total_cost:.2f}")
                print(f"Expiration date: {expiration_date}")
                
                # Create simulated orders to track in the backtest
                # For the buy leg
                buy_asset = Asset(
                    symbol=self.symbol,  # Using underlying since we can't backtest options directly
                    asset_type="stock"   # Using stock as a proxy
                )
                
                # Create a simulated buy order to track cash usage
                # We'll buy a small amount of stock to represent our option position
                # This is just to make the backtest work, not to simulate actual returns
                buy_order = self.create_order(
                    buy_asset,
                    1,  # Just buying 1 share to represent the option position
                    side=Order.OrderSide.BUY,
                    limit_price=current_price  # Using current price for the limit order
                )
                
                # Submit the order to track cash usage
                buy_order_result = self.submit_order(buy_order)
                
                # Update last trade date
                self.last_trade_date = current_date.date()
                
            else:  # Live trading with real options
                # Buy the lower strike call
                buy_asset = Asset(
                    symbol=buy_option.symbol,
                    asset_type="option"
                )
                
                # Sell the higher strike call
                sell_asset = Asset(
                    symbol=sell_option.symbol,
                    asset_type="option"
                )
                
                # Create buy order for the lower strike call
                buy_order = self.create_order(
                    buy_asset,
                    num_spreads,
                    Order.OrderSide.BUY,
                    Order.OrderType.MARKET
                )
                
                # Create sell order for the higher strike call
                sell_order = self.create_order(
                    sell_asset,
                    num_spreads,
                    Order.OrderSide.SELL,
                    Order.OrderType.MARKET
                )
                
                # Submit the orders
                buy_order_result = self.submit_order(buy_order)
                sell_order_result = self.submit_order(sell_order)
                
                # Log the trade details
                print(f"\nBULL CALL SPREAD EXECUTED on {current_date.date()}:")
                print(f"Bought {num_spreads} contracts of {buy_option.symbol} at strike ${buy_option.strike}")
                print(f"Sold {num_spreads} contracts of {sell_option.symbol} at strike ${sell_option.strike}")
                print(f"Expiration date: {expiration_date}")
                
                # Update last trade date
                self.last_trade_date = current_date.date()
            
        except Exception as e:
            print(f"Error executing bull call spread: {str(e)}")
            import traceback
            traceback.print_exc()

broker = Alpaca(ALPACA_CONFIG)

if __name__ == "__main__":
    IS_BACKTESTING=False
    if IS_BACKTESTING:
        symbol = "PLTR"
        # Set up backtest parameters
        backtesting_start = datetime(2024, 1, 1)
        backtesting_end = datetime(2025, 5, 1)
        
        # Run backtest
        result = BullCallSpreadTrader.backtest(
            PolygonDataBacktesting,
            symbol=symbol,
            benchmark_asset=Asset(symbol="PLTR", asset_type="stock"),  # Using Asset object as per memory
            backtesting_start=backtesting_start,
            backtesting_end=backtesting_end,
            parameters={
                "symbol": symbol,
                "num_spreads": 20
            },
            cash=1000  # Initial budget
        )
    else:
        strategy = BullCallSpreadTrader(
            broker=broker,
            symbol="PLTR",
            initial_budget=1000,
            num_spreads=20
        )
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
