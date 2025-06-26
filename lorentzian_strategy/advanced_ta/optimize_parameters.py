"""
Parameter Optimization for Lorentzian Classification Strategy
============================================================

This script optimizes the parameters of the Lorentzian Classification strategy
to maximize returns, win rate, or other performance metrics.

Usage: python optimize_parameters.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import itertools
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import psutil  # For system monitoring

# Load environment variables
load_dotenv()

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our existing functions
from run_advanced_ta import (
    download_real_data, 
    calculate_performance_metrics, 
    display_performance_report,
    aggregate_minute_to_daily,
    aggregate_hour_to_daily,
    aggregate_intraday_to_daily
)

try:
    from classifier import (
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

class OptimizationConfig:
    """Configuration for parameter optimization"""
    
    def __init__(self):
        # Market data settings
        self.symbol = os.getenv('SYMBOL', 'TSLA')
        self.start_date = os.getenv('BACKTESTING_START', '2024-01-31')
        self.end_date = os.getenv('BACKTESTING_END', '2024-12-31')
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
        self.timeframe = os.getenv('DATA_TIMEFRAME', 'day').lower()
        self.aggregate_to_daily = os.getenv('AGGREGATE_TO_DAILY', 'true').lower() == 'true'
        
        # Optimization settings
        self.max_combinations = int(os.getenv('MAX_COMBINATIONS', '2000'))  # Significantly increased default
        self.use_parallel = os.getenv('USE_PARALLEL', 'true').lower() == 'true'
        
        # CPU usage control with safety limits
        n_jobs_env = os.getenv('N_JOBS', '').strip()
        if n_jobs_env:
            self.n_jobs = int(n_jobs_env)
        else:
            # Default to conservative setting (leave 4 cores free)
            self.n_jobs = max(1, mp.cpu_count() - 4)
        
        # Safety check: never use more than 75% of available cores
        max_safe_cores = max(1, int(mp.cpu_count() * 0.75))
        if self.n_jobs > max_safe_cores:
            print(f"⚠️  WARNING: N_JOBS={self.n_jobs} may overload system with {mp.cpu_count()} cores")
            print(f"   Reducing to safe limit: {max_safe_cores} cores")
            self.n_jobs = max_safe_cores
        
        # Optimization strategy
        self.optimize_for_return = os.getenv('OPTIMIZE_FOR_RETURN', 'false').lower() == 'true'
        
        # Expanded parameter ranges for more thorough optimization
        self.param_ranges = {
            # Core ML settings - expanded ranges
            'neighborsCount': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20],
            'maxBarsBack': [300, 500, 750, 1000, 1250, 1500, 1750, 2000],
            'useDynamicExits': [True, False],
            
            # RSI Feature parameters - more granular
            'rsi_period': [8, 10, 12, 14, 16, 18, 20, 22, 24],
            'rsi_smooth': [1, 2, 3, 4, 5],
            
            # Williams %R (WT) Feature parameters - expanded
            'wt_n1': [5, 6, 7, 8, 9, 10, 11, 12, 14],
            'wt_n2': [6, 8, 10, 11, 12, 14, 16, 18],
            
            # CCI Feature parameters - more options
            'cci_period': [8, 10, 12, 14, 16, 18, 20, 22],
            'cci_smooth': [1, 2, 3, 4, 5],
            
            # EMA/SMA Filter settings - more periods
            'useEmaFilter': [True, False],
            'emaPeriod': [50, 100, 150, 200, 250],
            'useSmaFilter': [True, False],
            'smaPeriod': [50, 100, 150, 200, 250],
            
            # Advanced filter settings - more thresholds
            'useVolatilityFilter': [True, False],
            'useRegimeFilter': [True, False],
            'useAdxFilter': [True, False],
            'regimeThreshold': [-0.2, -0.1, 0.0, 0.1, 0.2],
            'adxThreshold': [15, 20, 25, 30, 35],
            
            # Kernel filter settings - expanded ranges
            'useKernelSmoothing': [True, False],
            'lookbackWindow': [4, 6, 8, 10, 12, 14, 16],
            'relativeWeight': [4.0, 6.0, 8.0, 10.0, 12.0, 15.0],
            'regressionLevel': [15, 20, 25, 30, 35],
            'crossoverLag': [1, 2, 3, 4, 5],
        }
        
        # Optimization objectives - adjust based on strategy
        if self.optimize_for_return:
            # Prioritize total return heavily
            self.objectives = {
                'total_return': {'weight': 0.7, 'direction': 'maximize'},     # 70% - heavily prioritize return
                'win_rate': {'weight': 0.1, 'direction': 'maximize'},        # 10%
                'profit_factor': {'weight': 0.1, 'direction': 'maximize'},   # 10%
                'sharpe_ratio': {'weight': 0.05, 'direction': 'maximize'},   # 5%
                'max_drawdown': {'weight': 0.05, 'direction': 'minimize'},   # 5%
            }
        else:
            # Balanced approach (default)
            self.objectives = {
                'total_return': {'weight': 0.4, 'direction': 'maximize'},    # 40%
                'win_rate': {'weight': 0.2, 'direction': 'maximize'},        # 20%  
                'profit_factor': {'weight': 0.2, 'direction': 'maximize'},   # 20%
                'sharpe_ratio': {'weight': 0.1, 'direction': 'maximize'},    # 10%
                'max_drawdown': {'weight': 0.1, 'direction': 'minimize'},    # 10%
            }

def test_parameter_combination_wrapper(params, df, symbol, initial_capital, objectives):
    """Wrapper function for parallel processing"""
    result = test_parameter_combination(params, df, symbol, initial_capital)
    if result:
        result['optimization_score'] = calculate_optimization_score(result, objectives)
    return result

def test_parameter_combination(params, df, symbol, initial_capital):
    """Test a single parameter combination and return performance metrics"""
    try:
        # Create features with optimized parameters
        features = [
            Feature("RSI", params['rsi_period'], params['rsi_smooth']),
            Feature("WT", params['wt_n1'], params['wt_n2']),
            Feature("CCI", params['cci_period'], params['cci_smooth'])
        ]
        
        # Create settings (now with optimizable EMA/SMA filters)
        settings = Settings(
            source=df['close'],
            neighborsCount=params['neighborsCount'],
            maxBarsBack=params['maxBarsBack'],
            useDynamicExits=params['useDynamicExits'],
            useEmaFilter=params['useEmaFilter'],
            emaPeriod=params['emaPeriod'],
            useSmaFilter=params['useSmaFilter'],
            smaPeriod=params['smaPeriod']
        )
        
        # Create kernel filter (now with optimizable crossoverLag)
        kernel_filter = KernelFilter(
            useKernelSmoothing=params['useKernelSmoothing'],
            lookbackWindow=params['lookbackWindow'],
            relativeWeight=params['relativeWeight'],
            regressionLevel=params['regressionLevel'],
            crossoverLag=params['crossoverLag']
        )
        
        # Create filter settings (now with optimizable thresholds)
        filter_settings = FilterSettings(
            useVolatilityFilter=params['useVolatilityFilter'],
            useRegimeFilter=params['useRegimeFilter'],
            useAdxFilter=params['useAdxFilter'],
            regimeThreshold=params['regimeThreshold'],
            adxThreshold=params['adxThreshold'],
            kernelFilter=kernel_filter
        )
        
        # Run classification
        lc = LorentzianClassification(df, features, settings, filter_settings)
        results = lc.data
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(results, symbol, initial_capital)
        
        if metrics is None:
            return None
        
        # Add parameter info to metrics
        metrics['parameters'] = params.copy()
        
        return metrics
        
    except Exception as e:
        # More detailed error reporting for debugging
        import traceback
        error_details = str(e)
        if "division by zero" in error_details.lower():
            print(f"❌ Division by zero error in parameter test: {error_details}")
        elif "invalid value" in error_details.lower():
            print(f"❌ Invalid value error in parameter test: {error_details}")
        else:
            print(f"❌ Error testing parameters: {error_details}")
        # Uncomment below for full traceback during debugging
        # traceback.print_exc()
        return None

def calculate_optimization_score(metrics, objectives):
    """Calculate a composite optimization score based on multiple objectives"""
    if not metrics:
        return -float('inf')
    
    score = 0
    for metric_name, config in objectives.items():
        weight = config['weight']
        direction = config['direction']
        
        if metric_name in metrics:
            value = metrics[metric_name]
            
            # Normalize values to 0-1 range for scoring
            if metric_name == 'total_return':
                normalized_value = max(0, min(1, (value + 50) / 200))  # -50% to 150% range
            elif metric_name == 'win_rate':
                normalized_value = value / 100  # 0% to 100%
            elif metric_name == 'profit_factor':
                normalized_value = max(0, min(1, value / 5))  # 0 to 5 range
            elif metric_name == 'sharpe_ratio':
                normalized_value = max(0, min(1, (value + 2) / 4))  # -2 to 2 range
            elif metric_name == 'max_drawdown':
                normalized_value = max(0, min(1, (value + 50) / 50))  # -50% to 0% range
            else:
                normalized_value = 0.5  # Default neutral score
            
            if direction == 'minimize':
                normalized_value = 1 - normalized_value
            
            score += weight * normalized_value
    
    return score

def generate_parameter_combinations(config):
    """Generate parameter combinations using smart sampling strategies"""
    param_names = list(config.param_ranges.keys())
    param_values = list(config.param_ranges.values())
    
    # Calculate total possible combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"📊 Total possible combinations: {total_combinations:,}")
    
    if total_combinations <= config.max_combinations:
        # Use all combinations if feasible
        print(f"✅ Using all {total_combinations:,} combinations")
        all_combinations = list(itertools.product(*param_values))
    else:
        # Use smart sampling strategies
        print(f"🎯 Using smart sampling: {config.max_combinations:,} from {total_combinations:,} possible")
        
        # Strategy 1: Latin Hypercube Sampling for better coverage
        lhs_combinations = generate_latin_hypercube_sample(config.param_ranges, config.max_combinations // 2)
        
        # Strategy 2: Memory-efficient random sampling
        np.random.seed(42)  # For reproducibility
        n_random = config.max_combinations - len(lhs_combinations)
        random_sample = []
        
        for _ in range(n_random):
            combination = []
            for values in param_values:
                combination.append(np.random.choice(values))
            random_sample.append(tuple(combination))
        
        # Combine both strategies
        all_combinations = lhs_combinations + random_sample
        
        print(f"   📐 Latin Hypercube: {len(lhs_combinations):,} combinations")
        print(f"   🎲 Random sampling: {len(random_sample):,} combinations")
    
    # Convert to parameter dictionaries
    param_combinations = []
    for combination in all_combinations:
        params = dict(zip(param_names, combination))
        param_combinations.append(params)
    
    return param_combinations

def generate_latin_hypercube_sample(param_ranges, n_samples):
    """Generate Latin Hypercube Sample for better parameter space coverage"""
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Create LHS samples
    np.random.seed(42)
    n_params = len(param_names)
    
    # Generate LHS samples in [0,1] space
    samples = np.random.rand(n_samples, n_params)
    
    # Apply Latin Hypercube constraint
    for i in range(n_params):
        # Create permutation for this parameter
        perm = np.random.permutation(n_samples)
        # Scale to proper intervals
        samples[:, i] = (perm + samples[:, i]) / n_samples
    
    # Map to actual parameter values
    combinations = []
    for sample in samples:
        combination = []
        for i, (param_name, values) in enumerate(zip(param_names, param_values)):
            # Map [0,1] to parameter index
            idx = int(sample[i] * len(values))
            if idx >= len(values):
                idx = len(values) - 1
            combination.append(values[idx])
        combinations.append(tuple(combination))
    
    return combinations

def display_optimization_summary(results, config):
    """Display a summary of optimization results"""
    if not results:
        print("❌ No valid results found")
        return
    
    print("\n" + "="*80)
    print("🔧 PARAMETER OPTIMIZATION SUMMARY")
    print("="*80)
    
    print(f"📊 Symbol: {config.symbol}")
    print(f"📅 Period: {config.start_date} to {config.end_date}")
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"🔄 Combinations Tested: {len(results)}")
    
    # Sort by optimization score
    results.sort(key=lambda x: x['optimization_score'], reverse=True)
    
    print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS:")
    print("="*80)
    
    for i, result in enumerate(results[:5], 1):
        metrics = result
        params = result['parameters']
        
        print(f"\n#{i} - Optimization Score: {result['optimization_score']:.3f}")
        print("-" * 60)
        
        # Key performance metrics
        total_return_color = "🟢" if metrics['total_return'] > 0 else "🔴"
        win_rate_color = "🟢" if metrics['win_rate'] >= 50 else "🟡" if metrics['win_rate'] >= 40 else "🔴"
        
        print(f"{total_return_color} Total Return: {metrics['total_return']:+7.2f}% | Final Value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"{win_rate_color} Win Rate: {metrics['win_rate']:5.1f}% | Trades: {metrics['total_trades']} | Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"⚡ Sharpe: {metrics['sharpe_ratio']:5.2f} | Max DD: {metrics['max_drawdown']:6.2f}% | Avg P&L: ${metrics['avg_dollar_return_per_trade']:+,.0f}")
        
        # Key parameters
        print(f"🔧 Neighbors: {params['neighborsCount']} | Bars Back: {params['maxBarsBack']} | Dynamic Exits: {params['useDynamicExits']}")
        print(f"📈 RSI({params['rsi_period']},{params['rsi_smooth']}) | WT({params['wt_n1']},{params['wt_n2']}) | CCI({params['cci_period']},{params['cci_smooth']})")
        print(f"📊 EMA({params['emaPeriod']})={params['useEmaFilter']} | SMA({params['smaPeriod']})={params['useSmaFilter']}")
        print(f"🎛️  Filters: Vol={params['useVolatilityFilter']} | Regime={params['useRegimeFilter']} | ADX={params['useAdxFilter']}")
        print(f"⚙️  Kernel: Smooth={params['useKernelSmoothing']} | Window={params['lookbackWindow']} | Weight={params['relativeWeight']} | Lag={params['crossoverLag']}")
    
    # Best parameters
    best_result = results[0]
    print(f"\n🎯 BEST PARAMETERS (Score: {best_result['optimization_score']:.3f}):")
    print("="*80)
    
    best_params = best_result['parameters']
    for param, value in best_params.items():
        print(f"   {param}: {value}")

def save_optimization_results(results, config):
    """Save optimization results to JSON file"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(results_dir, f"optimization_results_{config.symbol}_{timestamp}.json")
    
    # Prepare data for JSON serialization
    json_data = {
        'config': {
            'symbol': config.symbol,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'initial_capital': config.initial_capital,
            'max_combinations': config.max_combinations,
            'param_ranges': config.param_ranges,
            'objectives': config.objectives
        },
        'results': []
    }
    
    for result in results:
        # Convert numpy types to native Python types for JSON
        json_result = {}
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating)):
                json_result[key] = float(value)
            elif isinstance(value, dict):
                json_result[key] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                                  for k, v in value.items()}
            else:
                json_result[key] = value
        json_data['results'].append(json_result)
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"💾 Optimization results saved to: {filename}")
    return filename

def save_best_parameters(results, config):
    """Save the best parameters to a standardized file for easy loading"""
    if not results:
        return None
    
    # Get the best result
    best_result = max(results, key=lambda x: x['optimization_score'])
    best_params = best_result['parameters']
    
    # Create best parameters file with both run_advanced_ta.py and AdvancedLorentzianStrategy formats
    best_params_data = {
        'optimization_info': {
            'symbol': config.symbol,
            'optimization_date': datetime.now().isoformat(),
            'start_date': config.start_date,
            'end_date': config.end_date,
            'initial_capital': config.initial_capital,
            'optimization_score': best_result['optimization_score'],
            'total_return': best_result['total_return'],
            'win_rate': best_result['win_rate'],
            'total_trades': best_result['total_trades'],
            'final_portfolio_value': best_result['final_portfolio_value']
        },
        
        # Format for run_advanced_ta.py (LorentzianClassification)
        'best_parameters': {
            # Core settings
            'neighborsCount': best_params['neighborsCount'],
            'maxBarsBack': best_params['maxBarsBack'],
            'useDynamicExits': best_params['useDynamicExits'],
            
            # EMA/SMA Filter settings
            'useEmaFilter': best_params['useEmaFilter'],
            'emaPeriod': best_params['emaPeriod'],
            'useSmaFilter': best_params['useSmaFilter'],
            'smaPeriod': best_params['smaPeriod'],
            
            # Feature parameters
            'features': [
                {
                    'type': 'RSI',
                    'param1': best_params['rsi_period'],
                    'param2': best_params['rsi_smooth']
                },
                {
                    'type': 'WT',
                    'param1': best_params['wt_n1'],
                    'param2': best_params['wt_n2']
                },
                {
                    'type': 'CCI',
                    'param1': best_params['cci_period'],
                    'param2': best_params['cci_smooth']
                }
            ],
            
            # Filter settings
            'filter_settings': {
                'useVolatilityFilter': best_params['useVolatilityFilter'],
                'useRegimeFilter': best_params['useRegimeFilter'],
                'useAdxFilter': best_params['useAdxFilter'],
                'regimeThreshold': best_params['regimeThreshold'],
                'adxThreshold': best_params['adxThreshold']
            },
            
            # Kernel filter settings
            'kernel_filter': {
                'useKernelSmoothing': best_params['useKernelSmoothing'],
                'lookbackWindow': best_params['lookbackWindow'],
                'relativeWeight': best_params['relativeWeight'],
                'regressionLevel': best_params['regressionLevel'],
                'crossoverLag': best_params['crossoverLag']
            }
        },
        
        # Format for AdvancedLorentzianStrategy (Lumibot parameters dict)
        'lumibot_parameters': {
            'symbols': [config.symbol],
            'neighbors': best_params['neighborsCount'],
            'history_window': max(best_params['maxBarsBack'], 2000),
            'max_bars_back': best_params['maxBarsBack'],
            'use_dynamic_exits': best_params['useDynamicExits'],
            'use_ema_filter': best_params['useEmaFilter'],
            'ema_period': best_params['emaPeriod'],
            'use_sma_filter': best_params['useSmaFilter'],
            'sma_period': best_params['smaPeriod'],
            'use_volatility_filter': best_params['useVolatilityFilter'],
            'use_regime_filter': best_params['useRegimeFilter'],
            'use_adx_filter': best_params['useAdxFilter'],
            'regime_threshold': best_params['regimeThreshold'],
            'adx_threshold': best_params['adxThreshold'],
            'use_kernel_smoothing': best_params['useKernelSmoothing'],
            'kernel_lookback': best_params['lookbackWindow'],
            'kernel_weight': best_params['relativeWeight'],
            'regression_level': best_params['regressionLevel'],
            'crossover_lag': best_params['crossoverLag'],
            'features': [
                ('RSI', best_params['rsi_period'], best_params['rsi_smooth']),
                ('WT', best_params['wt_n1'], best_params['wt_n2']),
                ('CCI', best_params['cci_period'], best_params['cci_smooth'])
            ]
        }
    }
    
    # Save to standardized filename in the same directory as this script
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
    best_params_file = os.path.join(script_dir, f"best_parameters_{config.symbol}.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    best_params_data_converted = convert_numpy_types(best_params_data)
    
    with open(best_params_file, 'w') as f:
        json.dump(best_params_data_converted, f, indent=2)
    
    print(f"🎯 Best parameters saved to: {best_params_file}")
    print(f"   📋 Compatible with run_advanced_ta.py and AdvancedLorentzianStrategy")
    return best_params_file

def load_lumibot_parameters(symbol):
    """
    Load optimized parameters for AdvancedLorentzianStrategy (Lumibot)
    
    Usage in your strategy:
    from optimize_parameters import load_lumibot_parameters
    
    class MyStrategy(AdvancedLorentzianStrategy):
        def initialize(self):
            # Load optimized parameters
            optimized_params = load_lumibot_parameters('TSLA')
            if optimized_params:
                self.parameters.update(optimized_params)
            super().initialize()
    """
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
    best_params_file = os.path.join(script_dir, f"best_parameters_{symbol}.json")
    
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r') as f:
                data = json.load(f)
            
            if 'lumibot_parameters' in data:
                print(f"✅ Loaded optimized Lumibot parameters for {symbol}")
                print(f"   Expected return: {data['optimization_info']['total_return']:+.2f}%")
                print(f"   Expected win rate: {data['optimization_info']['win_rate']:.1f}%")
                return data['lumibot_parameters']
            else:
                print(f"⚠️  Old parameter format found for {symbol}, please re-run optimization")
                return None
                
        except Exception as e:
            print(f"❌ Error loading Lumibot parameters: {e}")
            return None
    else:
        print(f"ℹ️  No optimized parameters found for {symbol}")
        print(f"   Run 'python optimize_parameters.py' to generate optimized parameters")
        return None

def monitor_system_resources():
    """Monitor and display system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"💻 System Resources:")
    print(f"   CPU Usage: {cpu_percent:.1f}%")
    print(f"   Memory Usage: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
    print(f"   Available Memory: {memory.available // (1024**3):.1f}GB")
    
    # Warning if system is already under load
    if cpu_percent > 50:
        print(f"⚠️  WARNING: CPU already at {cpu_percent:.1f}% - consider reducing N_JOBS")
    if memory.percent > 80:
        print(f"⚠️  WARNING: Memory at {memory.percent:.1f}% - optimization may be slow")

def main():
    """Main optimization function"""
    print("🔧 Starting Parameter Optimization for Lorentzian Classification")
    print("="*80)
    
    # Check system resources before starting
    monitor_system_resources()
    print()
    
    # Load configuration
    config = OptimizationConfig()
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Date range: {config.start_date} to {config.end_date}")
    print(f"   Data timeframe: {config.timeframe}")
    print(f"   Initial capital: ${config.initial_capital:,.2f}")
    print(f"   Max combinations: {config.max_combinations:,}")
    print(f"   Parallel processing: {config.use_parallel}")
    if config.use_parallel:
        print(f"   CPU cores: {config.n_jobs}")
    
    # Display optimization strategy
    strategy_name = "Return-Focused" if config.optimize_for_return else "Balanced"
    print(f"\n🎯 Optimization Strategy: {strategy_name}")
    print(f"   Objective Weights:")
    for metric, obj in config.objectives.items():
        print(f"     • {metric}: {obj['weight']*100:.0f}% ({obj['direction']})")
    print()
    
    # Validate timeframe
    if config.timeframe not in ['day', 'hour', 'minute']:
        print(f"⚠️  Invalid timeframe '{config.timeframe}', defaulting to 'day'")
        config.timeframe = 'day'
    
    # Warning for intraday data optimization
    if config.timeframe in ['minute', 'hour']:
        start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        if config.timeframe == 'minute' and days_diff > 7:
            print(f"⚠️  WARNING: Optimizing with {days_diff} days of minute data!")
            print(f"   This will be VERY slow. Consider using daily data or shorter date range.")
            print(f"   Recommended: Use daily data (DATA_TIMEFRAME=day) for optimization")
        elif config.timeframe == 'hour' and days_diff > 90:
            print(f"⚠️  WARNING: Optimizing with {days_diff} days of hourly data!")
            print(f"   This may be slow. Consider using daily data for faster optimization.")
            print(f"   Recommended: Use daily data (DATA_TIMEFRAME=day) for optimization")
    
    # Download market data
    print(f"\n📥 Downloading market data...")
    df = download_real_data(symbol=config.symbol, start_date=config.start_date, end_date=config.end_date, timeframe=config.timeframe)
    
    # Handle intraday data aggregation for optimization
    if config.timeframe in ['minute', 'hour'] and config.aggregate_to_daily:
        print(f"\n📊 Aggregating {config.timeframe} data to daily for optimization...")
        df = aggregate_intraday_to_daily(df, config.timeframe)
        print(f"✅ Using daily aggregated data for faster optimization")
    
    # Convert to lowercase columns for classification
    df_for_classification = df.copy()
    df_for_classification.columns = df_for_classification.columns.str.lower()
    
    # Generate parameter combinations
    print(f"\n🔄 Generating parameter combinations...")
    param_combinations = generate_parameter_combinations(config)
    print(f"   Testing {len(param_combinations):,} parameter combinations")
    
    # Run optimization
    print(f"\n🧠 Running optimization...")
    results = []
    
    if config.use_parallel and len(param_combinations) > 10:
        # Parallel processing for large numbers of combinations
        print(f"⚡ Using parallel processing with {config.n_jobs} cores...")
        
        # Create partial function with fixed arguments
        test_func = partial(
            test_parameter_combination_wrapper,
            df=df_for_classification,
            symbol=config.symbol,
            initial_capital=config.initial_capital,
            objectives=config.objectives
        )
        
        # Use multiprocessing pool
        with mp.Pool(processes=config.n_jobs) as pool:
            # Use imap for progress tracking
            with tqdm(total=len(param_combinations), desc="Testing combinations") as pbar:
                for i, result in enumerate(pool.imap(test_func, param_combinations)):
                    if result:
                        results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        'Valid Results': len(results),
                        'CPU Cores': config.n_jobs,
                        'Success Rate': f"{len(results)/(i+1)*100:.1f}%" if i > 0 else "0%"
                    })
                    
                    # Periodic system monitoring (every 100 combinations)
                    if (i + 1) % 100 == 0:
                        cpu_now = psutil.cpu_percent(interval=0.1)
                        if cpu_now > 90:
                            print(f"\n⚠️  High CPU usage detected: {cpu_now:.1f}%")
                            print(f"   Consider stopping (Ctrl+C) and reducing N_JOBS")
    else:
        # Sequential processing for small numbers or when parallel is disabled
        print(f"🔄 Using sequential processing...")
        with tqdm(total=len(param_combinations), desc="Testing combinations") as pbar:
            for i, params in enumerate(param_combinations):
                result = test_parameter_combination(params, df_for_classification, config.symbol, config.initial_capital)
                if result:
                    result['optimization_score'] = calculate_optimization_score(result, config.objectives)
                    results.append(result)
                pbar.update(1)
                
                # Show progress every 10 combinations
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({'Valid Results': len(results)})
    
    print(f"\n✅ Optimization completed! Found {len(results)} valid results")
    
    if results:
        # Display results
        display_optimization_summary(results, config)
        
        # Save results
        save_optimization_results(results, config)
        
        # Save best parameters for easy loading
        best_params_file = save_best_parameters(results, config)
        
        # Show detailed report for best result
        best_result = max(results, key=lambda x: x['optimization_score'])
        print(f"\n🏆 DETAILED REPORT FOR BEST PARAMETERS:")
        display_performance_report(best_result)
        
    else:
        print("❌ No valid results found. Check your parameter ranges and data.")

if __name__ == "__main__":
    main() 