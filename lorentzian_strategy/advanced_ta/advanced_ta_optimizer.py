"""
Advanced TA Optimizer
====================

This optimizer uses the advanced_ta package to find optimal parameters for 
Lorentzian Classification that yield the highest returns.

The optimizer tests different combinations of:
- Feature parameters (RSI, WT, CCI, ADX lengths and smoothing)
- Classification settings (neighbors count, max bars back)
- Filter settings (volatility, regime, ADX filters)
- Kernel filter settings (lookback window, relative weight, regression level)

Returns the optimal parameter set that maximizes trading returns.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from polygon import RESTClient
import itertools
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lorentzian_classification import (
        LorentzianClassification, 
        Feature, 
        Settings, 
        FilterSettings, 
        KernelFilter,
        Direction
    )
    print("✅ Successfully imported LorentzianClassification components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the advanced_ta directory")
    sys.exit(1)

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    parameters: Dict
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float

class AdvancedTAOptimizer:
    """
    Optimizer for Advanced TA Lorentzian Classification parameters
    """
    
    def __init__(self, symbol: str = 'SPY', start_date: str = '2023-01-01', end_date: str = '2024-12-31'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
        # Define parameter ranges for optimization (REDUCED for reasonable testing)
        self.param_ranges = {
            # Feature parameters
            'rsi_length': [14, 21],
            'rsi_smooth': [1, 2],
            'wt_channel': [9, 11],
            'wt_average': [10, 12],
            'cci_length': [14, 20],
            'cci_smooth': [1, 2],
            'adx_length': [14, 20],
            
            # Classification settings
            'neighbors_count': [6, 8],
            'max_bars_back': [1500, 2000],
            'use_dynamic_exits': [False, True],
            
            # EMA/SMA Filter settings
            'use_ema_filter': [False, True],
            'ema_period': [100, 200],
            'use_sma_filter': [False, True],
            'sma_period': [100, 200],
            
            # Filter settings
            'use_volatility_filter': [False, True],
            'use_regime_filter': [False, True],
            'use_adx_filter': [False, True],
            'regime_threshold': [-0.1, 0.0],
            'adx_threshold': [15, 20],
            
            # Kernel filter settings
            'use_kernel_smoothing': [False, True],
            'lookback_window': [6, 8],
            'relative_weight': [6.0, 8.0],
            'regression_level': [20, 25],
            'crossover_lag': [1, 2]
        }
        
    def download_data(self) -> pd.DataFrame:
        """Download market data for optimization"""
        print(f"📥 Downloading data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        # Get API key from environment
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            print("❌ POLYGON_API_KEY not found in environment variables")
            return self._generate_sample_data()
        
        try:
            # Initialize Polygon client
            polygon_client = RESTClient(polygon_api_key)
            
            # Get aggregates from Polygon
            aggs = []
            for agg in polygon_client.get_aggs(
                ticker=self.symbol,
                multiplier=1,
                timespan="day",
                from_=self.start_date,
                to=self.end_date,
                limit=5000
            ):
                aggs.append(agg)
            
            if not aggs:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Convert to DataFrame
            data_list = []
            for agg in aggs:
                data_list.append({
                    'date': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data_list)
            df = df.sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Convert to lowercase columns (required by advanced_ta)
            df.columns = df.columns.str.lower()
            
            print(f"✅ Downloaded {len(df)} days of data")
            return df
            
        except Exception as e:
            print(f"❌ Failed to download data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, days: int = 500) -> pd.DataFrame:
        """Generate sample data as fallback"""
        print(f"📊 Generating {days} days of sample data...")
        
        np.random.seed(42)
        start_date = datetime.now() - timedelta(days=int(days * 1.4))
        dates = pd.date_range(start=start_date, periods=days, freq='B')
        
        initial_price = 200.0
        returns = np.random.normal(0.0005, 0.02, days)
        
        prices = [initial_price]
        for i in range(1, days):
            price = prices[-1] * (1 + returns[i])
            prices.append(price)
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_range = abs(np.random.normal(0, 0.015)) * close
            high = close + np.random.uniform(0, daily_range)
            low = close - np.random.uniform(0, daily_range)
            open_price = low + np.random.uniform(0, high - low)
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            volume = int(np.random.uniform(1000000, 10000000))
            
            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def create_features(self, params: Dict) -> List[Feature]:
        """Create feature list based on parameters"""
        features = [
            Feature("RSI", params['rsi_length'], params['rsi_smooth']),
            Feature("WT", params['wt_channel'], params['wt_average']),
            Feature("CCI", params['cci_length'], params['cci_smooth']),
            Feature("ADX", params['adx_length'], 2)  # ADX only uses one parameter
        ]
        return features
    
    def create_settings(self, params: Dict) -> Settings:
        """Create settings based on parameters"""
        settings = Settings(
            source=self.data['close'],
            neighborsCount=params['neighbors_count'],
            maxBarsBack=params['max_bars_back'],
            useDynamicExits=params['use_dynamic_exits'],
            useEmaFilter=params['use_ema_filter'],
            emaPeriod=params['ema_period'],
            useSmaFilter=params['use_sma_filter'],
            smaPeriod=params['sma_period']
        )
        return settings
    
    def create_filter_settings(self, params: Dict) -> FilterSettings:
        """Create filter settings based on parameters"""
        kernel_filter = KernelFilter(
            useKernelSmoothing=params['use_kernel_smoothing'],
            lookbackWindow=params['lookback_window'],
            relativeWeight=params['relative_weight'],
            regressionLevel=params['regression_level'],
            crossoverLag=params['crossover_lag']
        )
        
        filter_settings = FilterSettings(
            useVolatilityFilter=params['use_volatility_filter'],
            useRegimeFilter=params['use_regime_filter'],
            useAdxFilter=params['use_adx_filter'],
            regimeThreshold=params['regime_threshold'],
            adxThreshold=params['adx_threshold'],
            kernelFilter=kernel_filter
        )
        return filter_settings
    
    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics from classification results"""
        # Get trading signals
        long_signals = results_df['startLongTrade'].notna()
        short_signals = results_df['startShortTrade'].notna()
        
        # Calculate returns for each trade
        returns = []
        current_position = None
        entry_price = None
        
        for i, row in results_df.iterrows():
            if pd.notna(row['startLongTrade']) and current_position != 'long':
                if current_position == 'short' and entry_price is not None:
                    # Close short position
                    trade_return = (entry_price - row['close']) / entry_price
                    returns.append(trade_return)
                # Open long position
                current_position = 'long'
                entry_price = row['close']
                
            elif pd.notna(row['startShortTrade']) and current_position != 'short':
                if current_position == 'long' and entry_price is not None:
                    # Close long position
                    trade_return = (row['close'] - entry_price) / entry_price
                    returns.append(trade_return)
                # Open short position
                current_position = 'short'
                entry_price = row['close']
        
        # Close final position if needed
        if current_position is not None and entry_price is not None:
            final_price = results_df['close'].iloc[-1]
            if current_position == 'long':
                trade_return = (final_price - entry_price) / entry_price
            else:  # short
                trade_return = (entry_price - final_price) / entry_price
            returns.append(trade_return)
        
        if not returns:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'profit_factor': 0.0
            }
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = np.prod(1 + returns) - 1
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate Sharpe ratio (assuming 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(returns),
            'avg_trade_return': np.mean(returns),
            'profit_factor': profit_factor
        }
    
    def evaluate_parameters(self, params: Dict) -> OptimizationResult:
        """Evaluate a parameter set and return performance metrics"""
        try:
            # Create components
            features = self.create_features(params)
            settings = self.create_settings(params)
            filter_settings = self.create_filter_settings(params)
            
            # Run classification
            lc = LorentzianClassification(self.data, features, settings, filter_settings)
            results_df = lc.data
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(results_df)
            
            return OptimizationResult(
                parameters=params,
                **metrics
            )
            
        except Exception as e:
            print(f"❌ Error evaluating parameters: {e}")
            return OptimizationResult(
                parameters=params,
                total_return=-999.0,
                win_rate=0.0,
                max_drawdown=-1.0,
                sharpe_ratio=-999.0,
                total_trades=0,
                avg_trade_return=0.0,
                profit_factor=0.0
            )
    
    def optimize(self, max_combinations: int = 1000, optimization_metric: str = 'total_return') -> List[OptimizationResult]:
        """
        Run optimization to find best parameters
        
        Args:
            max_combinations: Maximum number of parameter combinations to test
            optimization_metric: Metric to optimize ('total_return', 'sharpe_ratio', 'profit_factor')
        """
        print(f"🎯 Starting Advanced TA Optimization for {self.symbol}")
        print(f"   Date range: {self.start_date} to {self.end_date}")
        print(f"   Optimization metric: {optimization_metric}")
        
        # Download data
        self.data = self.download_data()
        
        # Calculate theoretical maximum combinations (for information)
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        
        total_possible = 1
        for values in param_values:
            total_possible *= len(values)
        
        print(f"📊 Parameter space:")
        print(f"   Total possible combinations: {total_possible:,}")
        print(f"   Testing {max_combinations:,} combinations ({max_combinations/total_possible*100:.6f}%)")
        
        # Generate random parameter combinations (MEMORY EFFICIENT)
        print(f"🔍 Generating and testing parameter combinations...")
        
        np.random.seed(42)  # For reproducibility
        results = []
        tested_combinations = set()  # Track tested to avoid duplicates
        
        for i in range(max_combinations):
            # Generate a single random combination
            combination = []
            for values in param_values:
                combination.append(np.random.choice(values))
            
            # Convert to tuple for hashing (to check duplicates)
            combination_tuple = tuple(combination)
            
            # Skip if already tested
            if combination_tuple in tested_combinations:
                continue
                
            tested_combinations.add(combination_tuple)
            
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Show progress
            if i % 50 == 0:
                print(f"   Progress: {i}/{max_combinations} ({i/max_combinations*100:.1f}%)")
            
            # Evaluate parameters
            result = self.evaluate_parameters(params)
            results.append(result)
            
            # Show best so far
            if result.total_return > -900:  # Valid result
                current_best = max([r for r in results if r.total_return > -900], 
                                 key=lambda x: getattr(x, optimization_metric), default=None)
                if current_best and result == current_best:
                    print(f"   🚀 New best! {optimization_metric}: {getattr(result, optimization_metric):.4f}")
        
        # Sort results by optimization metric
        valid_results = [r for r in results if r.total_return > -900]
        valid_results.sort(key=lambda x: getattr(x, optimization_metric), reverse=True)
        
        print(f"\n✅ Optimization completed!")
        print(f"   Valid results: {len(valid_results)}/{len(results)}")
        print(f"   Unique combinations tested: {len(tested_combinations)}")
        
        return valid_results
    
    def print_results(self, results: List[OptimizationResult], top_n: int = 5):
        """Print optimization results"""
        print(f"\n📊 Top {min(top_n, len(results))} Parameter Sets:")
        print("=" * 80)
        
        for i, result in enumerate(results[:top_n]):
            print(f"\n{i+1}. Total Return: {result.total_return:.2%}")
            print(f"   Win Rate: {result.win_rate:.2%}")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {result.max_drawdown:.2%}")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            print(f"   Parameters:")
            
            # Group parameters by category
            feature_params = {k: v for k, v in result.parameters.items() 
                            if any(x in k for x in ['rsi', 'wt', 'cci', 'adx'])}
            classification_params = {k: v for k, v in result.parameters.items() 
                                   if any(x in k for x in ['neighbors', 'bars', 'dynamic', 'ema', 'sma'])}
            filter_params = {k: v for k, v in result.parameters.items() 
                           if any(x in k for x in ['volatility', 'regime', 'threshold'])}
            kernel_params = {k: v for k, v in result.parameters.items() 
                           if any(x in k for x in ['kernel', 'lookback', 'weight', 'regression', 'crossover'])}
            
            if feature_params:
                print(f"     Features: {feature_params}")
            if classification_params:
                print(f"     Classification: {classification_params}")
            if filter_params:
                print(f"     Filters: {filter_params}")
            if kernel_params:
                print(f"     Kernel: {kernel_params}")
    
    def save_results(self, results: List[OptimizationResult], filename: str = None):
        """Save optimization results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_results_{self.symbol}_{timestamp}.json"
        
        # Convert results to dictionaries and handle numpy types
        results_data = []
        for result in results:
            result_dict = asdict(result)
            
            # Convert numpy types to native Python types for JSON serialization
            for key, value in result_dict.items():
                if hasattr(value, 'item'):  # numpy scalar
                    result_dict[key] = value.item()
                elif isinstance(value, dict):
                    # Handle nested dictionaries (parameters)
                    for k, v in value.items():
                        if hasattr(v, 'item'):  # numpy scalar
                            value[k] = v.item()
            
            results_data.append(result_dict)
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'symbol': self.symbol,
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'results': results_data
                }, f, indent=2)
            
            print(f"💾 Results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print(f"💡 Results data types: {type(results_data[0]) if results_data else 'No results'}")
            return None


def main():
    """Main optimization function"""
    # Get configuration from environment
    symbol = os.getenv('SYMBOL', 'SPY')
    start_date = os.getenv('BACKTESTING_START', '2023-01-01')
    end_date = os.getenv('BACKTESTING_END', '2024-12-31')
    
    # Create optimizer
    optimizer = AdvancedTAOptimizer(symbol=symbol, start_date=start_date, end_date=end_date)
    
    # Run optimization
    results = optimizer.optimize(max_combinations=50, optimization_metric='total_return')
    
    # Print results
    optimizer.print_results(results, top_n=5)
    
    # Save results
    optimizer.save_results(results)
    
    # Return best parameters
    if results:
        best_result = results[0]
        print(f"\n🏆 BEST PARAMETERS FOR {symbol}:")
        print(f"   Expected Return: {best_result.total_return:.2%}")
        print(f"   Parameters: {best_result.parameters}")
        return best_result.parameters
    else:
        print("❌ No valid results found")
        return None


def run_comprehensive_optimization():
    """Run optimization with different configurations"""
    
    print("🚀 Advanced TA Lorentzian Classification Optimizer")
    print("=" * 60)
    
    # Get configuration
    symbol = os.getenv('SYMBOL', 'SPY')
    start_date = os.getenv('BACKTESTING_START', '2023-01-01')
    end_date = os.getenv('BACKTESTING_END', '2024-12-31')
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Date Range: {start_date} to {end_date}")
    print(f"   API Key: {'✅ Set' if os.getenv('POLYGON_API_KEY') else '❌ Missing'}")
    
    # Create optimizer
    optimizer = AdvancedTAOptimizer(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Run optimization with different metrics
    print(f"\n🎯 Running optimization...")
    
    # Option 1: Optimize for total return (default)
    print(f"\n1️⃣ Optimizing for Total Return...")
    results_return = optimizer.optimize(
        max_combinations=50,  # MUCH smaller for testing
        optimization_metric='total_return'
    )
    
    if results_return:
        print(f"\n📈 Best Parameters for Total Return:")
        optimizer.print_results(results_return, top_n=3)
        
        # Save results
        filename_return = optimizer.save_results(results_return, f"optimization_return_{symbol}.json")
        
        # Option 2: Optimize for Sharpe ratio
        print(f"\n2️⃣ Optimizing for Sharpe Ratio...")
        results_sharpe = optimizer.optimize(
            max_combinations=50,
            optimization_metric='sharpe_ratio'
        )
        
        if results_sharpe:
            print(f"\n📊 Best Parameters for Sharpe Ratio:")
            optimizer.print_results(results_sharpe, top_n=3)
            
            # Save results
            filename_sharpe = optimizer.save_results(results_sharpe, f"optimization_sharpe_{symbol}.json")
        
        # Option 3: Optimize for profit factor
        print(f"\n3️⃣ Optimizing for Profit Factor...")
        results_pf = optimizer.optimize(
            max_combinations=50,
            optimization_metric='profit_factor'
        )
        
        if results_pf:
            print(f"\n💰 Best Parameters for Profit Factor:")
            optimizer.print_results(results_pf, top_n=3)
            
            # Save results
            filename_pf = optimizer.save_results(results_pf, f"optimization_profit_factor_{symbol}.json")
        
        # Summary
        print(f"\n🏆 OPTIMIZATION SUMMARY FOR {symbol}")
        print("=" * 60)
        
        if results_return:
            best_return = results_return[0]
            print(f"🎯 Best Total Return: {best_return.total_return:.2%}")
            print(f"   Parameters: {best_return.parameters}")
            
        if results_sharpe:
            best_sharpe = results_sharpe[0]
            print(f"\n📊 Best Sharpe Ratio: {best_sharpe.sharpe_ratio:.2f}")
            print(f"   Total Return: {best_sharpe.total_return:.2%}")
            print(f"   Parameters: {best_sharpe.parameters}")
            
        if results_pf:
            best_pf = results_pf[0]
            print(f"\n💰 Best Profit Factor: {best_pf.profit_factor:.2f}")
            print(f"   Total Return: {best_pf.total_return:.2%}")
            print(f"   Parameters: {best_pf.parameters}")
        
        print(f"\n💾 Results saved to JSON files for further analysis")
        
        # Return the best overall result (highest return)
        return results_return[0] if results_return else None
        
    else:
        print("❌ No valid optimization results found")
        return None


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "comprehensive":
            # Run comprehensive optimization
            best_params = run_comprehensive_optimization()
        elif command == "simple":
            # Run simple optimization
            best_params = main()
        elif command == "help":
            print("🔧 Advanced TA Optimizer Usage:")
            print("  python advanced_ta_optimizer.py                 - Run simple optimization")
            print("  python advanced_ta_optimizer.py simple          - Run simple optimization") 
            print("  python advanced_ta_optimizer.py comprehensive   - Run comprehensive optimization")
            print("  python advanced_ta_optimizer.py help            - Show this help")
            print("\n📊 Environment Variables:")
            print("  SYMBOL                - Stock symbol (default: SPY)")
            print("  BACKTESTING_START     - Start date (default: 2023-01-01)")
            print("  BACKTESTING_END       - End date (default: 2024-12-31)")
            print("  POLYGON_API_KEY       - Your Polygon API key")
            sys.exit(0)
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python advanced_ta_optimizer.py help' for usage information")
            sys.exit(1)
    else:
        # Default to simple optimization
        best_params = main()
    
    if best_params:
        print(f"\n✅ Optimization completed successfully!")
        print(f"Use the best parameters in your trading strategy for optimal returns.")
    else:
        print(f"\n❌ Optimization failed. Check your data and parameters.") 