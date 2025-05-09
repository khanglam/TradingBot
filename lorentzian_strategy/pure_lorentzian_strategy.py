import datetime as dt
import pandas as pd
import ta
import numpy as np
import pytz
import sys

# Add the path to the advanced_ta package
sys.path.append('d:\\Khang\\Projects\\TradingViewWorkspace\\LorentzianClassification')

from lumibot.strategies.strategy import Strategy
from lumibot.credentials import ALPACA_CONFIG
from lumibot.backtesting import PolygonDataBacktesting

# Import the Lorentzian Classification components
from advanced_ta.LorentzianClassification.Classifier import LorentzianClassification
from advanced_ta.LorentzianClassification.Types import Feature, Settings, FilterSettings, KernelFilter, Direction


class PureLorentzianStrategy(Strategy):
    """
    Strategy that uses the Lorentzian Classification algorithm to generate trading signals.
    This implementation provides intelligent signal generation without any hardcoded logic.
    """
    
    def initialize(self):
        """Initialize the strategy"""
        self.sleeptime = "1D"  # Run the strategy once per day
        self.classifier = None  # Will be initialized on first trading iteration
        self.last_classification_time = None  # Track when we last classified data
        self.last_signals = {'buy': False, 'sell': False}  # Track last signals
        self.signal_history = []  # Track signal history for debugging
        print("Strategy initialized successfully")
        
    def on_trading_iteration(self):
        """Main trading logic executed on each iteration"""
        # Get current datetime and symbol
        current_dt = self.get_datetime()
        symbol = self.parameters.get("symbol", "SPY")
        
        # Log current info with reliable print statements for debugging
        print(f"TRADING ITERATION: {current_dt}, Symbol: {symbol}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}, Cash: ${self.cash:.2f}")
        
        # Get current position
        position = self.get_position(symbol)
        print(f"Current position: {position}")
        
        # Get the current price
        price = self.get_last_price(symbol)
        print(f"Last price of {symbol}: {price}")
        
        # Get trading parameters
        max_bars_back = self.parameters.get("max_bars_back", 300)
        position_size = self.parameters.get("position_size", 0.1)  # Default to 10% of cash
        
        # Update the classifier - this will also fetch historical data
        classification_success = self._update_classifier()
        
        # Signal generation - use the classifier
        buy_signal = False
        sell_signal = False
        
        # Extract signals from classifier if available
        if classification_success and self.classifier and hasattr(self.classifier, 'data') and not self.classifier.data.empty:
            buy_signal, sell_signal = self._extract_signals_from_classifier()
            self.last_signals = {'buy': buy_signal, 'sell': sell_signal}
            self.signal_history.append({
                'datetime': current_dt,
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'price': price
            })
            print(f"Signals from classifier - Buy: {buy_signal}, Sell: {sell_signal}")
        else:
            print("Classification unsuccessful, no signals generated")
            return  # Skip this iteration if we couldn't get signals
        
        # TRADING LOGIC - Based on signals with reliable execution
        # BUY LOGIC
        if position is None and buy_signal:
            print(f"BUY SIGNAL detected")
            
            # Calculate quantity based on position size parameter (with fallback)
            try:
                if price <= 0:
                    print(f"Invalid price of {price}, skipping trade")
                    return
                else:
                    # Calculate based on position size but use a minimum of 1 share
                    raw_quantity = (self.cash * position_size) / price 
                    quantity = max(1, int(raw_quantity))  # Ensure at least 1 share
                    print(f"Calculated quantity: {quantity} shares based on position size: {position_size*100}%")
            except Exception as e:
                print(f"Error calculating quantity: {e}, skipping trade")
                import traceback
                print(traceback.format_exc())
                return
            
            print(f"Creating BUY order for {quantity} shares of {symbol} at approximately ${price}")
            
            try:
                # Create order
                order = self.create_order(symbol, quantity, "buy")
                print(f"Order created: {order}")
                
                # Submit order
                result = self.submit_order(order)
                print(f"Order submitted with result: {result}")
            except Exception as e:
                print(f"ERROR submitting buy order: {e}")
                import traceback
                print(traceback.format_exc())
        
        # SELL LOGIC
        elif position is not None and sell_signal:
            print(f"SELL SIGNAL detected for existing position")
            quantity = position.quantity
            
            try:
                # Create order
                order = self.create_order(symbol, quantity, "sell")
                print(f"Sell order created: {order}")
                
                # Submit order
                result = self.submit_order(order)
                print(f"Sell order submitted with result: {result}")
            except Exception as e:
                print(f"ERROR submitting sell order: {e}")
                import traceback
                print(traceback.format_exc())
        
    def _update_classifier(self):
        """Update the classifier with the latest data"""
        # Get parameters
        symbol = self.parameters["symbol"]
        max_bars_back = self.parameters.get("max_bars_back", 300)
        
        # Get historical data as DataFrame
        df = self._get_historical_data()
        
        # Check if we have data
        if df is None or len(df) < 50:
            print("Not enough historical data for classification")
            return False
                
        # Initialize classifier if needed
        if self.classifier is None:
            print("Initializing Lorentzian classifier")
            try:
                features = self._build_features()
                settings = self._build_settings()
                filter_settings = self._build_filter_settings()
                self.classifier = LorentzianClassification(df, features, settings, filter_settings)
                print("Classifier initialized successfully")
            except Exception as e:
                print(f"Error initializing classifier: {e}")
                import traceback
                print(traceback.format_exc())
                return False
        else:
            # Update existing classifier with new data
            try:
                self.classifier.update_data(df)
                print("Classifier data updated successfully")
            except Exception as e:
                print(f"Error updating classifier data: {e}")
                return False
        
        # Classify the data
        try:
            self.classifier.classify()
            print("Classification completed")
            self.last_classification_time = self.get_datetime()
            return True
        except Exception as e:
            print(f"Error during classification: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def _build_features(self):
        """Build features for the classifier from parameters"""
        features_param = self.parameters.get("features", None)
        
        if features_param is not None:
            # Convert parameter features to Feature objects
            features = []
            for feature_dict in features_param:
                feature_type = feature_dict.get("type", "RSI")
                param1 = feature_dict.get("param1", 14)
                param2 = feature_dict.get("param2", 2)
                features.append(Feature(feature_type, param1, param2))
            return features
        else:
            # Default features if none provided
            return [
                Feature("RSI", 14, 2),
                Feature("WT", 10, 11),
                Feature("CCI", 20, 2),
                Feature("ADX", 20, 2)
            ]
    
    def _build_settings(self):
        """Build settings for the classifier"""
        neighbors_count = self.parameters.get("neighbors_count", 8)
        max_bars_back = self.parameters.get("max_bars_back", 300)
        use_kernel_smoothing = self.parameters.get("use_kernel_smoothing", True)
        
        return Settings(
            neighbors_count=neighbors_count,
            max_bars_back=max_bars_back,
            use_kernel_smoothing=use_kernel_smoothing
        )
    
    def _build_filter_settings(self):
        """Build filter settings for the classifier"""
        use_volatility_filter = self.parameters.get("use_volatility_filter", False)
        use_regime_filter = self.parameters.get("use_regime_filter", False)
        use_adx_filter = self.parameters.get("use_adx_filter", False)
        
        return FilterSettings(
            use_volatility_filter=use_volatility_filter,
            use_regime_filter=use_regime_filter,
            use_adx_filter=use_adx_filter
        )
    
    def _get_historical_data(self):
        """Get historical data and convert to DataFrame"""
        symbol = self.parameters["symbol"]
        max_bars_back = self.parameters.get("max_bars_back", 300)
        
        # Get historical data
        print(f"Getting historical data for {symbol}, {max_bars_back} bars")
        bars = self.get_historical_prices(symbol, max_bars_back, "day")
        
        # Check if we have data
        if bars is None:
            print("No historical data available")
            return None
            
        # Convert bars to DataFrame
        try:
            # Try to convert bars to a DataFrame
            if hasattr(bars, 'df'):
                # Newer versions of Lumibot have a df attribute
                df = bars.df.copy()
                # Ensure column names are lowercase
                if 'open' not in df.columns and 'Open' in df.columns:
                    df.columns = [col.lower() for col in df.columns]
            else:
                # Otherwise, construct DataFrame from the list of bars
                df = pd.DataFrame({
                    'open': [bar.open for bar in bars],
                    'high': [bar.high for bar in bars],
                    'low': [bar.low for bar in bars],
                    'close': [bar.close for bar in bars],
                    'volume': [bar.volume for bar in bars]
                }, index=[bar.timestamp for bar in bars])
            
            return df
        except Exception as e:
            print(f"Error converting bars to DataFrame: {e}")
            return None
            
    def _extract_signals_from_classifier(self):
        """Extract trading signals from the classifier data"""
        if not self.classifier or not hasattr(self.classifier, 'data') or self.classifier.data.empty:
            print("No classifier data available for signal extraction")
            return False, False  # No buy or sell signals
            
        try:
            # Get the latest row of data
            latest_data = self.classifier.data.iloc[-1]
            
            # Initialize signals
            buy_signal = False
            sell_signal = False
            
            # Check for signals based on the available columns
            if 'signal' in latest_data:
                buy_signal = latest_data['signal'] == 1
                sell_signal = latest_data['signal'] == -1
            elif 'startLongTrade' in latest_data:
                buy_signal = not pd.isna(latest_data['startLongTrade'])
                sell_signal = not pd.isna(latest_data['startShortTrade'])
            elif 'direction' in latest_data:
                buy_signal = latest_data['direction'] == Direction.LONG
                sell_signal = latest_data['direction'] == Direction.SHORT
            
            # Add indicators to the chart if available
            if hasattr(self.classifier, 'yhat1') and hasattr(self.classifier, 'yhat2'):
                self.add_line("Kernel Regression", self.classifier.yhat1[-1])
                self.add_line("Kernel Smoothed", self.classifier.yhat2[-1])
                
            return buy_signal, sell_signal
            
        except Exception as e:
            print(f"Error extracting signals from classifier: {e}")
            import traceback
            print(traceback.format_exc())
            return False, False  # Return no signals on error
    
    def get_signal_history(self):
        """Return the signal history for analysis"""
        return pd.DataFrame(self.signal_history)


if __name__ == "__main__":
    # Import API keys from config
    try:
        from config import ALPACA_CONFIG
    except ImportError:
        print("Could not import ALPACA_CONFIG from config.py")
        sys.exit(1)
        
    # Set up backtesting parameters
    tzinfo = pytz.timezone('America/New_York')
    backtesting_start = tzinfo.localize(dt.datetime(2024, 1, 1))
    backtesting_end = tzinfo.localize(dt.datetime(2024, 3, 31))
    
    # Additional backtesting parameters
    timestep = 'day'
    refresh_cache = True  # Force refresh of data cache
    
    # Run backtest with pure Lorentzian signals
    results, strategy = PureLorentzianStrategy.run_backtest(
        datasource_class=PolygonDataBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        minutes_before_closing=0,
        benchmark_asset='SPY',
        analyze_backtest=True,
        parameters={
            "symbol": "SPY",
            "max_bars_back": 300,
            "neighbors_count": 8,
            "use_dynamic_exits": True,
            "features": [
                {"type": "RSI", "param1": 7, "param2": 2},
                {"type": "WT", "param1": 10, "param2": 11},
                {"type": "CCI", "param1": 20, "param2": 2},
                {"type": "ADX", "param1": 14, "param2": 2},
            ],
            "use_volatility_filter": True,
            "use_regime_filter": False,
            "use_adx_filter": True,
            "use_kernel_smoothing": True,
            "position_size": 0.2,
        },
        show_progress_bar=True,
        
        # PolygonDataBacktesting kwargs
        config=ALPACA_CONFIG,
        timestep=timestep,
        refresh_cache=refresh_cache,
    )
    
    # Print the results
    print(results)
    
    # Print signal history if available
    if hasattr(strategy, 'get_signal_history') and callable(strategy.get_signal_history):
        signal_history = strategy.get_signal_history()
        if not signal_history.empty:
            print("\nSignal History:")
            print(signal_history)
