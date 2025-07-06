"""
Advanced Lorentzian Classification Strategy
==========================================

This strategy integrates the standardized LorentzianClassification from classifier.py
into a Lumibot-compatible strategy. It ensures consistency with optimization scripts.

Key improvements:
- Uses the same classifier.py as test_parameters.py and optimize_parameters.py
- Eliminates algorithm discrepancies between optimization and strategy execution
- Standardized data structures and signal processing
- Compatible with optimized parameters from optimize_parameters.py

Based on the research by @jdehorty on TradingView and enhanced for Lumibot.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from lumibot.credentials import IS_BACKTESTING

# Import the standardized classifier components
from classifier import (
    LorentzianClassification, 
    Feature, 
    Settings, 
    FilterSettings, 
    KernelFilter,
    Direction
)

# Load environment variables
load_dotenv()

class AdvancedLorentzianStrategy(Strategy):
    """
    Advanced Lorentzian Classification Strategy for Lumibot
    
    This strategy uses the standardized LorentzianClassification from classifier.py
    to ensure perfect consistency with optimization results from optimize_parameters.py.
    
    Key features:
    - Uses EXACT same algorithm as optimization scripts
    - Automatic parameter loading from best_parameters_{symbol}.json
    - Compatible with all classifier.py features and filters
    - Consistent signal processing and data handling
    
    Set environment variable USE_OPTIMIZED_PARAMS=false to disable optimization.
    """
    
    parameters = {
        'symbols': [os.getenv('SYMBOL', 'TSLA')],  # Read from .env file
        'neighbors': 8,
        'history_window': 2000,
        'max_bars_back': 500,
        'use_dynamic_exits': False,
        'use_ema_filter': False,
        'ema_period': 200,
        'use_sma_filter': False,
        'sma_period': 200,
        'use_volatility_filter': True,
        'use_regime_filter': True,
        'use_adx_filter': False,
        'regime_threshold': -0.1,
        'adx_threshold': 20,
        'use_kernel_smoothing': False,
        'kernel_lookback': 8,
        'kernel_weight': 8.0,
        'regression_level': 25,
        'crossover_lag': 2,
        'features': [
            ('RSI', 14, 2),
            ('WT', 10, 11),
            ('CCI', 20, 2),
            ('ADX', 20, 2),
            ('RSI', 9, 2)
        ]
    }
    
    def load_optimized_parameters(self, symbol):
        """Load optimized parameters from JSON file if available"""
        try:
            # Look for the file in the same directory as this script
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
            best_params_file = os.path.join(script_dir, f"best_parameters_{symbol}.json")
            
            print(f"🔍 Checking for optimized parameters...")
            print(f"   Looking for: {best_params_file}")
            
            if os.path.exists(best_params_file):
                print(f"✅ Found optimization file for {symbol}")
                
                with open(best_params_file, 'r') as f:
                    data = json.load(f)
                
                if 'best_parameters' in data:
                    print(f"🎯 Loading optimized parameters for {symbol}")
                    print(f"   📅 Optimization date: {data['optimization_info']['optimization_date'][:10]}")
                    print(f"   📈 Expected return: {data['optimization_info']['total_return']:+.2f}%")
                    print(f"   🎯 Expected win rate: {data['optimization_info']['win_rate']:.1f}%")
                    print(f"   🔢 Total trades: {data['optimization_info']['total_trades']}")
                    print(f"   💰 Final portfolio value: ${data['optimization_info']['final_portfolio_value']:,.2f}")
                    
                    return data['best_parameters']
                else:
                    print(f"⚠️  Optimization file found but missing 'best_parameters' key")
                    return None
            else:
                print(f"ℹ️  No optimization file found for {symbol}")
                print(f"   Run 'python optimize_parameters.py' to generate optimized parameters")
                return None
                
        except Exception as e:
            print(f"❌ Error loading optimized parameters: {str(e)}")
            return None
    
    def create_features_from_params(self, params):
        """Create Feature objects from parameter data"""
        if params is None:
            # Default features if no optimized parameters
            return [
                Feature("RSI", 14, 2),
                Feature("WT", 10, 11),
                Feature("CCI", 20, 2),
                Feature("ADX", 20, 2),
                Feature("RSI", 9, 2)
            ]
        
        features = []
        for feature_data in params['features']:
            features.append(Feature(
                feature_data['type'],
                feature_data['param1'],
                feature_data['param2']
            ))
        
        return features
    
    def create_settings_from_params(self, params, df_source):
        """Create Settings object from parameter data"""
        if params is None:
            # Default settings if no optimized parameters
            return Settings(
                source=df_source,
                neighborsCount=8,
                maxBarsBack=2000,
                useDynamicExits=False,
                useEmaFilter=False,
                emaPeriod=200,
                useSmaFilter=False,
                smaPeriod=200
            )
        
        return Settings(
            source=df_source,
            neighborsCount=params['neighborsCount'],
            maxBarsBack=params['maxBarsBack'],
            useDynamicExits=params['useDynamicExits'],
            useEmaFilter=params['useEmaFilter'],
            emaPeriod=params['emaPeriod'],
            useSmaFilter=params['useSmaFilter'],
            smaPeriod=params['smaPeriod']
        )
    
    def create_filter_settings_from_params(self, params):
        """Create FilterSettings object from parameter data"""
        if params is None:
            # Default filter settings if no optimized parameters
            kernel_filter = KernelFilter(
                useKernelSmoothing=False,
                lookbackWindow=8,
                relativeWeight=8.0,
                regressionLevel=25,
                crossoverLag=2
            )
            
            return FilterSettings(
                useVolatilityFilter=True,
                useRegimeFilter=True,
                useAdxFilter=False,
                regimeThreshold=-0.1,
                adxThreshold=20,
                kernelFilter=kernel_filter
            )
        
        # Create kernel filter from parameters
        kernel_filter = KernelFilter(
            useKernelSmoothing=params['kernel_filter']['useKernelSmoothing'],
            lookbackWindow=params['kernel_filter']['lookbackWindow'],
            relativeWeight=params['kernel_filter']['relativeWeight'],
            regressionLevel=params['kernel_filter']['regressionLevel'],
            crossoverLag=params['kernel_filter']['crossoverLag']
        )
        
        return FilterSettings(
            useVolatilityFilter=params['filter_settings']['useVolatilityFilter'],
            useRegimeFilter=params['filter_settings']['useRegimeFilter'],
            useAdxFilter=params['filter_settings']['useAdxFilter'],
            regimeThreshold=params['filter_settings']['regimeThreshold'],
            adxThreshold=params['filter_settings']['adxThreshold'],
            kernelFilter=kernel_filter
        )
    
    def initialize(self):
        """Initialize the strategy with optimized parameters"""
        self.sleeptime = "1D"  # Run once per day
        
        # Get the primary symbol
        self.symbols = self.parameters.get('symbols', [os.getenv('SYMBOL', 'TSLA')])
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
        
        symbol = self.symbols[0]  # Use first symbol for parameter loading
        
        # Try to load optimized parameters
        use_optimized = os.getenv('USE_OPTIMIZED_PARAMS', 'true').lower() == 'true'
        optimized_params = None
        param_source = "DEFAULT"
        
        if use_optimized:
            optimized_params = self.load_optimized_parameters(symbol)
            if optimized_params:
                param_source = "OPTIMIZED"
        
        # Create classifier components using standardized classifier.py classes
        self.features = self.create_features_from_params(optimized_params)
        
        # Note: Settings will be created per iteration with actual data source
        # FilterSettings can be created once
        self.filter_settings = self.create_filter_settings_from_params(optimized_params)
        
        # Store parameters for later use
        self.optimized_params = optimized_params
        
        # Log configuration
        print(f"⚙️  STRATEGY CONFIGURATION ({param_source} PARAMETERS)")
        print("-" * 60)
        print(f"🧠 ML Settings:")
        if optimized_params:
            print(f"   • Neighbors: {optimized_params.get('neighborsCount', 8)}")
            print(f"   • Max bars back: {optimized_params.get('maxBarsBack', 2000)}")
            print(f"   • Dynamic exits: {optimized_params.get('useDynamicExits', False)}")
        else:
            print(f"   • Neighbors: 8 (default)")
            print(f"   • Max bars back: 2000 (default)")
            print(f"   • Dynamic exits: False (default)")
        
        print(f"📊 Filters:")
        print(f"   • Volatility: {self.filter_settings.useVolatilityFilter}")
        print(f"   • Regime: {self.filter_settings.useRegimeFilter} (threshold: {self.filter_settings.regimeThreshold})")
        print(f"   • ADX: {self.filter_settings.useAdxFilter} (threshold: {self.filter_settings.adxThreshold})")
        print(f"   • Kernel smoothing: {self.filter_settings.kernelFilter.useKernelSmoothing}")
        
        print(f"📈 Features: {[(f.type, f.param1, f.param2) for f in self.features]}")
        print("="*80)
        print()
        
        # Initialize tracking variables
        if not hasattr(self.vars, 'position_state'):
            self.vars.position_state = {symbol: 'neutral' for symbol in self.symbols}
        if not hasattr(self.vars, 'last_signal'):
            self.vars.last_signal = {symbol: 0 for symbol in self.symbols}
    
    def on_trading_iteration(self):
        """Main trading logic executed on each iteration"""
        for symbol in self.symbols:
            try:
                asset = Asset(symbol, Asset.AssetType.STOCK)
                
                # Get historical data - use same logic as AdvancedLorentzianStrategy
                history_length = 2000  # Default from classifier.py Settings
                if self.optimized_params:
                    history_length = max(
                        self.optimized_params.get('maxBarsBack', 2000),
                        self.optimized_params.get('history_window', 2000)
                    )
                
                bars = self.get_historical_prices(asset, history_length, "day")
                
                if bars is None or len(bars.df) < 100:
                    self.log_message(f"Insufficient data for {symbol}", color="yellow")
                    continue
                
                df = bars.df.copy()
                
                # Ensure we have required columns and convert to lowercase (classifier.py format)
                if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    self.log_message(f"Missing required columns for {symbol}", color="red")
                    continue
                
                # Convert to lowercase columns as expected by classifier.py
                df.columns = df.columns.str.lower()
                
                # Create Settings with actual data source
                settings = self.create_settings_from_params(self.optimized_params, df['close'])
                
                # Run standardized LorentzianClassification (same as optimization scripts)
                try:
                    lc = LorentzianClassification(df, self.features, settings, self.filter_settings)
                    
                    # Get latest signals from classifier data
                    if len(lc.data) == 0:
                        self.log_message(f"No classification results for {symbol}", color="yellow")
                        continue
                    
                    # Extract signals using classifier.py format
                    latest_data = lc.data.iloc[-1]
                    
                    # Check for different signal column formats from classifier.py
                    if 'isNewBuySignal' in lc.data.columns and 'isNewSellSignal' in lc.data.columns:
                        # Boolean format (most common in classifier.py)
                        start_long = latest_data['isNewBuySignal']
                        start_short = latest_data['isNewSellSignal']
                    elif 'startLongTrade' in lc.data.columns and 'startShortTrade' in lc.data.columns:
                        # Price format
                        start_long = not pd.isna(latest_data['startLongTrade'])
                        start_short = not pd.isna(latest_data['startShortTrade'])
                    else:
                        # Fallback - no signals
                        start_long = False
                        start_short = False
                    
                    current_price = latest_data['close']
                    
                    # Get signal direction if available
                    current_signal = 0
                    if 'signal' in lc.data.columns:
                        current_signal = latest_data['signal']
                    elif start_long:
                        current_signal = Direction.LONG
                    elif start_short:
                        current_signal = Direction.SHORT
                    
                    # Log signal information
                    if current_signal != self.vars.last_signal.get(symbol, 0):
                        signal_name = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(current_signal, "UNKNOWN")
                        self.log_message(f"{symbol}: Signal changed to {signal_name} at ${current_price:.2f}", 
                                       color="blue")
                        self.vars.last_signal[symbol] = current_signal
                    
                    # Get current position
                    position = self.get_position(asset)
                    current_qty = position.quantity if position else 0
                    
                    # Execute EXACT same trading logic as AdvancedLorentzianSimulator
                    if start_long and current_qty <= 0:
                        # Enter long position
                        if current_qty < 0:
                            # Close short position first
                            close_order = self.create_order(asset, abs(current_qty), Order.OrderSide.BUY)
                            self.submit_order(close_order)
                            self.log_message(f"Closed SHORT position: {abs(current_qty)} {symbol} @ ${current_price:.2f}", 
                                           color="orange")
                        
                        # Open long position - EXACT same sizing as AdvancedLorentzianSimulator
                        cash = self.get_cash()
                        position_size = min(cash * 0.95, cash - 1000)  # EXACT formula from simulator
                        qty = int(position_size / current_price)
                        
                        if qty > 0:
                            buy_order = self.create_order(asset, qty, Order.OrderSide.BUY)
                            self.submit_order(buy_order)
                            self.log_message(f"Opened LONG position: {qty} {symbol} @ ${current_price:.2f}", 
                                           color="green")
                            self.vars.position_state[symbol] = 'long'
                    
                    elif start_short and current_qty >= 0:
                        # Enter short position (close long only - no actual short selling)
                        if current_qty > 0:
                            # Close long position
                            close_order = self.create_order(asset, current_qty, Order.OrderSide.SELL)
                            self.submit_order(close_order)
                            self.log_message(f"Closed LONG position: {current_qty} {symbol} @ ${current_price:.2f}", 
                                           color="orange")
                            self.vars.position_state[symbol] = 'neutral'
                        
                        # Note: AdvancedLorentzianSimulator does NOT open short positions
                        # This matches the exact behavior from the simulator
                    
                    # Add charting
                    self.add_line(f"{symbol}_price", current_price, color="black")
                    
                    # Add signal markers
                    if start_long:
                        self.add_marker(f"{symbol}_long", current_price, color="green", 
                                      symbol="triangle-up", size=10)
                    elif start_short:
                        self.add_marker(f"{symbol}_short", current_price, color="red", 
                                      symbol="triangle-down", size=10)
                
                except Exception as classification_error:
                    self.log_message(f"Classification error for {symbol}: {str(classification_error)}", color="red")
                    continue
                
            except Exception as e:
                self.log_message(f"Error processing {symbol}: {str(e)}", color="red")
                continue
    
    def on_abrupt_closing(self):
        """Handle strategy shutdown gracefully"""
        self.log_message("Strategy shutting down, closing all positions...", color="yellow")
        for symbol in self.symbols:
            try:
                asset = Asset(symbol, Asset.AssetType.STOCK)
                position = self.get_position(asset)
                if position and position.quantity != 0:
                    side = Order.OrderSide.SELL if position.quantity > 0 else Order.OrderSide.BUY
                    order = self.create_order(asset, abs(position.quantity), side)
                    self.submit_order(order)
            except Exception as e:
                self.log_message(f"Error closing position for {symbol}: {str(e)}", color="red")

# =====================================
# ============= RUNNER ================
# =====================================

if __name__ == "__main__":
    # Override parameters as needed
    strategy_params = AdvancedLorentzianStrategy.parameters.copy()
    
    if IS_BACKTESTING:
        # Backtesting configuration
        trading_fee = TradingFee(percent_fee=0.001)  # 0.1% trading fee
        
        AdvancedLorentzianStrategy.backtest(
            datasource_class=PolygonDataBacktesting,
            benchmark_asset=Asset(os.getenv('SYMBOL', 'TSLA'), Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=strategy_params
        )
    else:
        # Live trading configuration
        trader = Trader()
        strategy = AdvancedLorentzianStrategy(
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=strategy_params,
        )
        trader.add_strategy(strategy)
        trader.run_all() 