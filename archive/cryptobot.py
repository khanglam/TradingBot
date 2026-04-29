from lumibot.backtesting import PolygonDataBacktesting
from lumibot.credentials import IS_BACKTESTING
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order
from lumibot.brokers import Alpaca
from datetime import datetime, timedelta
from lumibot.example_strategies import ccxt_backtesting_example

import os  # Import os to load environment variables
import pandas as pd
from pandas_ta import atr, rsi  # For technical indicators
from pandas import DataFrame, Timedelta

# Load environment variables
CASH_AT_RISK = float(os.getenv('CASH_AT_RISK', 0.25))
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'kraken')

class CryptoTrader(Strategy):
    def initialize(self, asset=None, cash_at_risk=0.25, window=21):
        if asset is None:
            raise ValueError("You must provide a valid asset pair")
        # For crypto, market is 24/7
        self.set_market("24/7")
        self.sleeptime = "1D"  # Daily trading
        self.asset = asset
        self.base, self.quote = asset
        self.window = window
        self.symbol = f"{self.base.symbol}/{self.quote.symbol}"
        self.positions_data = {}  # To track entry prices and stop/target levels
        self.last_trade = None
        self.order_quantity = 0.0
        self.cash_at_risk = cash_at_risk

    def _position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(asset=self.asset, quote=self.quote)
        if last_price is None:
            return cash, last_price, 0.0
        quantity = round(cash * self.cash_at_risk / last_price, 6)  # More precision for crypto
        return cash, last_price, quantity
        
    def _get_historical_prices(self):
        return self.get_historical_prices(asset=self.asset, length=self.window,
                                    timestep="day", quote=self.quote).df
                                    
    def get_sma(self, length=200):
        # Request more days to ensure enough valid data for SMA calculation
        bars = self.get_historical_prices(asset=self.asset, length=length+10, 
                                         timestep="day", quote=self.quote)
        if bars is None:
            self.log_message(f'Historical data unavailable for {self.symbol}')
            return None

        # Convert the historical data into a DataFrame
        df = bars.df
        if df.shape[0] < length:
            self.log_message(f'Not enough historical data to compute {length}-day SMA.')
            return None

        # Calculate the SMA using closing prices
        sma = df['close'].tail(length).mean()
        self.log_message(f'Calculated {length}-day SMA: {sma:.2f}')
        return sma

    def calculate_atr(self, length=14):
        bars = self.get_historical_prices(asset=self.asset, length=length+1, 
                                         timestep="day", quote=self.quote)
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
        bars = self.get_historical_prices(asset=self.asset, length=length+1, 
                                         timestep="day", quote=self.quote)
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

    # def _get_bbands(self, history_df=None):
    #     """Calculate Bollinger Bands for the asset"""
    #     if history_df is None:
    #         history_df = self._get_historical_prices()
            
    #     num_std_dev = 2.0
    #     close = 'close'

    #     df = DataFrame(index=history_df.index)
    #     df[close] = history_df[close]
    #     df['bbm'] = df[close].rolling(window=self.window).mean()
    #     df['bbu'] = df['bbm'] + df[close].rolling(window=self.window).std() * num_std_dev
    #     df['bbl'] = df['bbm'] - df[close].rolling(window=self.window).std() * num_std_dev
    #     df['bbb'] = (df['bbu'] - df['bbl']) / df['bbm']
    #     df['bbp'] = (df[close] - df['bbl']) / (df['bbu'] - df['bbl'])
    #     return df

    def on_trading_iteration(self):
        # During the backtest, we get the current time with self.get_datetime()
        current_dt = self.get_datetime()
        self.log_message(f"Trading iteration at {current_dt}")
        
        # Get position sizing information
        cash, last_price, quantity = self._position_sizing()
        if last_price is None:
            self.log_message("Could not get last price, skipping iteration")
            return
            
        # Get historical data and technical indicators
        # history_df = self._get_historical_prices()
        # bbands = self._get_bbands(history_df)
        sma200 = self.get_sma(length=200)
        rsi_value = self.calculate_rsi(length=14)
        atr_value = self.calculate_atr(length=14)
        
        if sma200 is None or pd.isna(rsi_value) or atr_value is None or pd.isna(atr_value):
            self.log_message("Missing technical indicators, skipping iteration")
            return
            
        # Get current position and holdings
        # For crypto, we need to use a different approach to get position
        position = self.get_positions()
        current_holdings = 0
        for pos in position:
            if pos.asset == self.base:
                current_holdings = pos.quantity
                break
        self.log_message(f"Current holdings of {self.base.symbol}: {current_holdings}")
        
        # Default to 2% of price if ATR calculation fails
        if atr_value is None or pd.isna(atr_value):
            atr_value = last_price * 0.02
            
        # Get Bollinger Band Percentage for signal generation
        # try:
        #     prev_bbp = bbands[bbands.index < current_dt].tail(1).bbp.values[0]
        # except (IndexError, KeyError):
        #     self.log_message("Could not get BBP value, skipping iteration")
        #     return
            
        # Check for take profit or stop loss if we have a position
        if current_holdings > 0 and self.symbol in self.positions_data:
            position_data = self.positions_data[self.symbol]
            
            # Take profit condition
            if last_price >= position_data["take_profit"]:
                try:
                    sell_order = self.create_order(self.base,
                                                current_holdings,
                                                side=Order.OrderSide.SELL,
                                                type=Order.OrderType.MARKET,
                                                quote=self.quote)
                    self.submit_order(sell_order)
                    profit = (last_price - position_data["entry_price"]) * current_holdings
                    self.log_message(f"TAKE PROFIT: Selling {current_holdings} units at {last_price:.2f}")
                    self.log_message(f"Entry: {position_data['entry_price']:.2f}, Profit: {profit:.2f}")
                    del self.positions_data[self.symbol]
                    self.last_trade = Order.OrderSide.SELL
                    self.order_quantity = 0.0
                    return
                except Exception as e:
                    self.log_message(f"Error executing take profit: {e}")
                
            # Stop loss condition
            if last_price <= position_data["stop_loss"]:
                try:
                    sell_order = self.create_order(self.base,
                                                current_holdings,
                                                side=Order.OrderSide.SELL,
                                                type=Order.OrderType.MARKET,
                                                quote=self.quote)
                    self.submit_order(sell_order)
                    loss = (position_data["entry_price"] - last_price) * current_holdings
                    self.log_message(f"STOP LOSS: Selling {current_holdings} units at {last_price:.2f}")
                    self.log_message(f"Entry: {position_data['entry_price']:.2f}, Loss: {loss:.2f}")
                    del self.positions_data[self.symbol]
                    self.last_trade = Order.OrderSide.SELL
                    self.order_quantity = 0.0
                    return
                except Exception as e:
                    self.log_message(f"Error executing stop loss: {e}")
        
        # BUY SIGNAL - Oversold condition (BBP < -0.13) or RSI < 30
        if (rsi_value < 30) and cash > 0 and self.last_trade != Order.OrderSide.BUY and quantity > 0.0:
            # Set stop loss at 2 ATR below entry price
            stop_loss = last_price - (2 * atr_value)
            take_profit = last_price + (3 * atr_value)  # 3:2 reward-to-risk ratio
            
            # Store position data for tracking
            self.positions_data[self.symbol] = {
                "entry_price": last_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "quantity": quantity,
                "entry_date": current_dt
            }

            # Execute buy order
            order = self.create_order(self.base,
                                    quantity,
                                    side=Order.OrderSide.BUY,
                                    type=Order.OrderType.MARKET,
                                    quote=self.quote)
            self.submit_order(order)
            self.last_trade = Order.OrderSide.BUY
            self.order_quantity = quantity
            self.log_message(f"BUY ORDER: {quantity} units of {self.symbol} at {last_price:.2f}")
            self.log_message(f"Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            self.log_message(f"Risk: {(last_price - stop_loss) * quantity:.2f}, Reward: {(take_profit - last_price) * quantity:.2f}")

        # SELL SIGNAL - Overbought condition (BBP > 1.2) or RSI > 70
        elif (rsi_value > 70) and self.last_trade != Order.OrderSide.SELL and current_holdings > 0:
            order = self.create_order(self.base,
                                    current_holdings,
                                    side=Order.OrderSide.SELL,
                                    type=Order.OrderType.MARKET,
                                    quote=self.quote)
            self.submit_order(order)
            
            # Calculate profit/loss if we have the entry data
            if self.symbol in self.positions_data:
                entry_price = self.positions_data[self.symbol]["entry_price"]
                profit = (last_price - entry_price) * current_holdings
                self.log_message(f"SELL SIGNAL: Selling {current_holdings} units at {last_price:.2f}")
                self.log_message(f"Entry: {entry_price:.2f}, P/L: {profit:.2f}")
                del self.positions_data[self.symbol]
            else:
                self.log_message(f"SELL SIGNAL: Selling {current_holdings} units at {last_price:.2f}")
                
            self.last_trade = Order.OrderSide.SELL
            self.order_quantity = 0.0

if __name__ == "__main__":
    IS_BACKTESTING = True
    
    # Define crypto asset pair
    base_symbol = "DOGE"
    quote_symbol = "USD"
    asset = (Asset(symbol=base_symbol, asset_type="crypto"),
            Asset(symbol=quote_symbol, asset_type="crypto"))
    
    if IS_BACKTESTING:
        # Run backtest
        results = CryptoTrader.backtest(
            PolygonDataBacktesting,
            benchmark_asset=Asset(symbol=base_symbol, asset_type="crypto"),
            quote_asset=Asset(symbol=quote_symbol, asset_type="crypto"),
            parameters={
                "asset": asset,
                "cash_at_risk": CASH_AT_RISK,
                "window": 21,
            }
        )
        
        # Print backtest results
        print("Backtest completed!")
    else:
        # Live trading setup with Alpaca
        alpaca_config = {
            "API_KEY": os.getenv("ALPACA_API_KEY"),
            "API_SECRET": os.getenv("ALPACA_API_SECRET"),
            "PAPER": os.getenv("ALPACA_IS_PAPER", "True").lower() == "true",
            "BASE_URL": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        }
        
        broker = Alpaca(alpaca_config)
        strategy = CryptoTrader(broker=broker, 
                              asset=asset,
                              cash_at_risk=CASH_AT_RISK,
                              window=21)
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
