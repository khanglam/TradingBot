"""
Parameter Optimization for Lorentzian Classification Strategy
============================================================

This script optimizes the parameters of the Lorentzian Classification strategy
to maximize returns, win rate, or other performance metrics.

Usage: python optimize_parameters.py

For different log levels, set LOG_LEVEL in your .env file or run:
LOG_LEVEL=DEBUG python optimize_parameters.py  # Shows detailed logs and errors
LOG_LEVEL=INFO python optimize_parameters.py   # Shows progress bars and summaries (default)
LOG_LEVEL=WARN python optimize_parameters.py   # Shows only warnings and errors

For randomization control:
python optimize_parameters.py                  # Random results each run (default)
RANDOM_SEED=42 python optimize_parameters.py   # Reproducible results (same each run)
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

# Logging level control
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

def is_debug():
    """Returns True if DEBUG level logging is enabled"""
    return LOG_LEVEL == 'DEBUG'

def is_info():
    """Returns True if INFO level logging is enabled (includes DEBUG)"""
    return LOG_LEVEL in ['DEBUG', 'INFO']

def is_warn():
    """Returns True if WARN level logging is enabled (includes all levels)"""
    return LOG_LEVEL in ['DEBUG', 'INFO', 'WARN']

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our existing functions
from test_parameters import (
    download_real_data, 
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
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the advanced_ta directory")
    sys.exit(1)

def get_next_file_number(directory: str, base_filename: str, extension: str) -> int:
    """
    Get the next available number for incremental file naming
    
    Args:
        directory: Directory to search for existing files
        base_filename: Base filename pattern (e.g., "optimization_results_TSLA")
        extension: File extension (e.g., ".json")
        
    Returns:
        Next available number (1 if no files exist)
    """
    if not os.path.exists(directory):
        return 1
    
    import glob
    pattern = os.path.join(directory, f"{base_filename}_*.{extension}")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract numbers from existing files
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove base filename and extension, extract number
        try:
            # Pattern: base_filename_NUMBER.extension
            number_part = filename.replace(f"{base_filename}_", "").replace(f".{extension}", "")
            if number_part.isdigit():
                numbers.append(int(number_part))
        except:
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def generate_incremental_filename(directory: str, base_filename: str, extension: str) -> str:
    """
    Generate an incremental filename (e.g., optimization_results_TSLA_1.json)
    
    Args:
        directory: Directory where file will be saved
        base_filename: Base filename pattern
        extension: File extension (without dot)
        
    Returns:
        Full path with incremental number
    """
    next_number = get_next_file_number(directory, base_filename, extension)
    filename = f"{base_filename}_{next_number}.{extension}"
    return os.path.join(directory, filename)

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
        
        # Walk-forward analysis settings
        self.use_walk_forward = os.getenv('USE_WALK_FORWARD', 'false').lower() == 'true'
        self.walk_forward_periods = int(os.getenv('WALK_FORWARD_PERIODS', '3'))  # Number of periods to test
        
        # Randomization settings
        self.random_seed = os.getenv('RANDOM_SEED')  # Set to a number for reproducible results, leave unset for random
        
        # Optimization strategy
        self.optimize_for_return = os.getenv('OPTIMIZE_FOR_RETURN', 'false').lower() == 'true'
        
        # Expanded parameter ranges for more thorough optimization
        self.param_ranges = {
            # Core ML settings - expanded ranges
            'neighborsCount': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30],  # Added higher values
            'maxBarsBack': [200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500],  # Added lower and higher values
            'useDynamicExits': [True, False],
            
            # RSI Feature parameters - significantly expanded
            'rsi_period': [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40],  # Extended to 40
            'rsi_smooth': [1, 2, 3, 4, 5, 6, 7, 8],  # Added higher smoothing values
            
            # Williams %R (WT) Feature parameters - expanded
            'wt_n1': [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20],  # Extended range
            'wt_n2': [4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 22, 25],   # Extended range
            
            # CCI Feature parameters - more options
            'cci_period': [6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30],  # Extended range
            'cci_smooth': [1, 2, 3, 4, 5, 6, 7, 8],  # Added higher smoothing values
            
            # EMA/SMA Filter settings - more periods
            'useEmaFilter': [True, False],
            'emaPeriod': [20, 30, 50, 75, 100, 150, 200, 250, 300],  # Added more variety
            'useSmaFilter': [True, False],
            'smaPeriod': [20, 30, 50, 75, 100, 150, 200, 250, 300],  # Added more variety
            
            # Advanced filter settings - more thresholds
            'useVolatilityFilter': [True, False],
            'useRegimeFilter': [True, False],
            'useAdxFilter': [True, False],
            'regimeThreshold': [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],  # Wider range
            'adxThreshold': [10, 15, 20, 25, 30, 35, 40, 45],  # Wider range
            
            # Kernel filter settings - expanded ranges
            'useKernelSmoothing': [True, False],
            'lookbackWindow': [2, 4, 6, 8, 10, 12, 14, 16, 20, 25],  # Extended range
            'relativeWeight': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],  # Extended range
            'regressionLevel': [10, 15, 20, 25, 30, 35, 40, 50],  # Extended range
            'crossoverLag': [1, 2, 3, 4, 5, 6, 8, 10],  # Extended range
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

def validate_parameters(params):
    """Validate parameter combination for obvious issues"""
    # Check for reasonable ranges
    if params['neighborsCount'] < 1 or params['neighborsCount'] > 50:
        return False, "Invalid neighborsCount"
    
    if params['maxBarsBack'] < 50 or params['maxBarsBack'] > 5000:
        return False, "Invalid maxBarsBack"
    
    # Check technical indicator parameters
    if params['rsi_period'] < 2 or params['rsi_period'] > 50:
        return False, "Invalid RSI period"
    
    if params['wt_n1'] < 2 or params['wt_n2'] < 2:
        return False, "Invalid WT parameters"
    
    if params['cci_period'] < 2 or params['cci_period'] > 50:
        return False, "Invalid CCI period"
    
    # Check filter parameters
    if params['emaPeriod'] < 5 or params['emaPeriod'] > 500:
        return False, "Invalid EMA period"
    
    if params['smaPeriod'] < 5 or params['smaPeriod'] > 500:
        return False, "Invalid SMA period"
    
    return True, "Valid"

def test_parameter_combination(params, df, symbol, initial_capital):
    """Test a single parameter combination and return performance metrics"""
    try:
        # Validate parameters first
        is_valid, reason = validate_parameters(params)
        if not is_valid:
            return None
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
        
        # Simulate trading strategy using EXACT AdvancedLorentzianStrategy logic
        from simulate_trade import run_advanced_lorentzian_simulation
        results = run_advanced_lorentzian_simulation(df, features, settings, filter_settings, initial_capital)
        metrics = results['metrics']
        
        # Add symbol to metrics for compatibility
        metrics['symbol'] = symbol
        
        # Convert Trade dataclass objects to dictionaries for compatibility
        if 'all_trades' in metrics:
            trade_dicts = []
            for trade in metrics['all_trades']:
                trade_dicts.append({
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'side': trade.side,
                    'return_pct': trade.return_pct,
                    'return_dollars': trade.return_dollars,
                    'days_held': trade.days_held,
                    'reason': trade.reason
                })
            metrics['all_trades'] = trade_dicts
        
        if metrics is None:
            return None
        
        # Add parameter info to metrics
        metrics['parameters'] = params.copy()
        
        return metrics
        
    except Exception as e:
        # Track error types for analysis
        error_details = str(e)
        
        # Count different error types (stored in global variable for multiprocessing)
        if not hasattr(test_parameter_combination, 'error_counts'):
            test_parameter_combination.error_counts = {}
        
        if "division by zero" in error_details.lower():
            error_type = "division_by_zero"
        elif "invalid value" in error_details.lower() or "nan" in error_details.lower():
            error_type = "invalid_values"
        elif "insufficient" in error_details.lower() or "not enough" in error_details.lower():
            error_type = "insufficient_data"
        elif "index" in error_details.lower() and "out of" in error_details.lower():
            error_type = "index_error"
        else:
            error_type = "other"
        
        test_parameter_combination.error_counts[error_type] = test_parameter_combination.error_counts.get(error_type, 0) + 1
        
        # Only print errors occasionally to avoid spam
        if test_parameter_combination.error_counts[error_type] <= 3 and is_info():
            print(f"❌ {error_type}: {error_details}")
        
        return None

def calculate_optimization_score(metrics, objectives):
    """Calculate a composite optimization score based on multiple objectives"""
    if not metrics:
        return -float('inf')
    
    # Check for minimum viability criteria first
    if metrics['total_trades'] < 2:
        return -float('inf')  # Need at least 2 trades for meaningful statistics
    
    score = 0
    for metric_name, config in objectives.items():
        weight = config['weight']
        direction = config['direction']
        
        if metric_name in metrics:
            value = metrics[metric_name]
            
            # Improved normalization with more realistic ranges
            if metric_name == 'total_return':
                # Penalize extreme returns (likely overfitting)
                if abs(value) > 200:  # Returns over 200% are suspicious
                    normalized_value = 0.1
                else:
                    normalized_value = max(0, min(1, (value + 100) / 300))  # -100% to 200% range
            elif metric_name == 'win_rate':
                # Penalize extreme win rates (likely overfitting)
                if value > 90:  # Win rates over 90% are suspicious
                    normalized_value = 0.5
                else:
                    normalized_value = value / 100  # 0% to 100%
            elif metric_name == 'profit_factor':
                # Cap profit factor to avoid overfitting to extreme values
                capped_value = min(value, 10)  # Cap at 10
                normalized_value = max(0, min(1, capped_value / 10))  # 0 to 10 range
            elif metric_name == 'sharpe_ratio':
                # More realistic Sharpe ratio range
                normalized_value = max(0, min(1, (value + 1) / 3))  # -1 to 2 range
            elif metric_name == 'max_drawdown':
                # Penalize large drawdowns more severely
                if value < -50:  # Drawdowns over 50% are very bad
                    normalized_value = 0
                else:
                    normalized_value = max(0, min(1, (value + 50) / 50))  # -50% to 0% range
            else:
                normalized_value = 0.5  # Default neutral score
            
            if direction == 'minimize':
                normalized_value = 1 - normalized_value
            
            score += weight * normalized_value
    
    # Add penalty for suspicious combinations (likely overfitting)
    total_return = metrics.get('total_return', 0)
    win_rate = metrics.get('win_rate', 0)
    total_trades = metrics.get('total_trades', 0)
    
    # Penalty for unrealistic performance
    if total_return > 150 and win_rate > 80:  # Too good to be true
        score *= 0.5
    
    # Penalty for too few trades (not enough data)
    if total_trades < 5:
        score *= 0.7
    
    # Bonus for reasonable, consistent performance
    if 10 <= total_return <= 50 and 45 <= win_rate <= 70 and total_trades >= 10:
        score *= 1.1  # Small bonus for realistic performance
    
    return score

def generate_parameter_combinations(config):
    """Generate parameter combinations using smart sampling strategies"""
    param_names = list(config.param_ranges.keys())
    param_values = list(config.param_ranges.values())
    
    # Calculate total possible combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    if is_info():
        print(f"📊 Total possible combinations: {total_combinations:,}")
    
    if total_combinations <= config.max_combinations:
        # Use all combinations if feasible
        if is_info():
            print(f"✅ Using all {total_combinations:,} combinations")
        all_combinations = list(itertools.product(*param_values))
    else:
        # Use smart sampling strategies
        if is_info():
            print(f"🎯 Using smart sampling: {config.max_combinations:,} from {total_combinations:,} possible")
        
        # Strategy 1: Latin Hypercube Sampling for better coverage
        lhs_combinations = generate_latin_hypercube_sample(config.param_ranges, config.max_combinations // 2)
        
        # Strategy 2: Memory-efficient random sampling
        # Use random seed only if RANDOM_SEED environment variable is set
        random_seed = os.getenv('RANDOM_SEED')
        if random_seed:
            np.random.seed(int(random_seed))
            if is_info():
                print(f"   🎲 Using fixed random seed: {random_seed} (for reproducibility)")
        else:
            # Use truly random seed based on current time
            import time
            seed = int(time.time() * 1000000) % 2**32
            np.random.seed(seed)
            if is_info():
                print(f"   🎲 Using random seed: {seed} (set RANDOM_SEED env var for reproducibility)")
        
        n_random = config.max_combinations - len(lhs_combinations)
        random_sample = []
        
        for _ in range(n_random):
            combination = []
            for values in param_values:
                combination.append(np.random.choice(values))
            random_sample.append(tuple(combination))
        
        # Combine both strategies
        all_combinations = lhs_combinations + random_sample
        
        if is_info():
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
    # Use the same random state as the main sampling for consistency
    # (seed is already set in the calling function)
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
        print(f"{win_rate_color} Win Rate: {metrics['win_rate']:5.1f}% | Trades: {metrics['total_trades']} | Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"⚡ Sharpe: {metrics.get('sharpe_ratio', 0):5.2f} | Max DD: {metrics.get('max_drawdown', 0):6.2f}% | Avg P&L: ${metrics.get('avg_dollar_return_per_trade', 0):+,.0f}")
        
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
    
    filename = generate_incremental_filename(results_dir, f"optimization_results_{config.symbol}", "json")
    
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

def load_existing_best_parameters(symbol):
    """Load existing best parameters if available"""
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
    best_params_file = os.path.join(script_dir, f"best_parameters_{symbol}.json")
    
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r') as f:
                data = json.load(f)
            
            if 'optimization_info' in data and 'optimization_score' in data['optimization_info']:
                print(f"📋 Found existing best parameters for {symbol}")
                print(f"   Previous best score: {data['optimization_info']['optimization_score']:.3f}")
                print(f"   Previous best return: {data['optimization_info']['total_return']:+.2f}%")
                print(f"   Optimization date: {data['optimization_info']['optimization_date'][:10]}")
                return data
            else:
                print(f"⚠️  Existing parameter file has old format, will be updated")
                return None
                
        except Exception as e:
            print(f"❌ Error loading existing parameters: {e}")
            return None
    else:
        print(f"ℹ️  No existing best parameters found for {symbol}")
        return None

def save_best_parameters(results, config, existing_best=None):
    """Save the best parameters to a standardized file for easy loading"""
    if not results:
        return None
    
    # Get the best result from current optimization
    current_best = max(results, key=lambda x: x['optimization_score'])
    
    # Compare with existing best if available
    if existing_best and 'optimization_info' in existing_best:
        existing_score = existing_best['optimization_info']['optimization_score']
        current_score = current_best['optimization_score']
        
        print(f"\n🏆 OPTIMIZATION COMPARISON:")
        print(f"   Current best score:  {current_score:.3f} (return: {current_best['total_return']:+.2f}%)")
        print(f"   Previous best score: {existing_score:.3f} (return: {existing_best['optimization_info']['total_return']:+.2f}%)")
        
        if current_score <= existing_score:
            print(f"   📊 RESULT: Previous parameters remain the best!")
            print(f"   🔒 No update needed - keeping existing best parameters")
            
            # Update the existing file with new optimization attempt info
            existing_best['optimization_info']['last_optimization_attempt'] = datetime.now().isoformat()
            existing_best['optimization_info']['attempts_count'] = existing_best['optimization_info'].get('attempts_count', 1) + 1
            
            # Save updated existing best
            script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
            best_params_file = os.path.join(script_dir, f"best_parameters_{config.symbol}.json")
            
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                else:
                    return obj
            
            existing_best_converted = convert_numpy_types(existing_best)
            
            with open(best_params_file, 'w') as f:
                json.dump(existing_best_converted, f, indent=2)
            
            print(f"   📝 Updated attempt count and timestamp")
            return best_params_file
        else:
            print(f"   🎉 NEW RECORD! Current optimization found better parameters!")
            print(f"   📈 Improvement: +{current_score - existing_score:.3f} score points")
            # Continue with saving new best parameters
    else:
        print(f"\n🎯 FIRST OPTIMIZATION: Setting baseline best parameters")
    
    # Use current best result
    best_result = current_best
    best_params = best_result['parameters']
    
    # Create best parameters file with both test_parameters.py and AdvancedLorentzianStrategy formats
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
            'final_portfolio_value': best_result['final_portfolio_value'],
            # Tracking info for absolute best across all runs
            'first_found_date': datetime.now().isoformat(),
            'last_optimization_attempt': datetime.now().isoformat(),
            'attempts_count': 1,
            'is_new_record': True
        },
        
        # Format for test_parameters.py (LorentzianClassification)
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
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    best_params_data_converted = convert_numpy_types(best_params_data)
    
    with open(best_params_file, 'w') as f:
        json.dump(best_params_data_converted, f, indent=2)
    
    print(f"🎯 Best parameters saved to: {best_params_file}")
    print(f"   📋 Compatible with test_parameters.py and AdvancedLorentzianStrategy")
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

def validate_optimization_vs_strategy(config):
    """
    Validate that optimization simulation matches AdvancedLorentzianStrategy logic
    This helps ensure optimization results translate to real strategy performance
    """
    print(f"\n🔍 STRATEGY COMPATIBILITY VALIDATION")
    print("="*60)
    
    # Force daily data for consistency with AdvancedLorentzianStrategy
    if config.timeframe != 'day':
        print(f"⚠️  WARNING: Data timeframe mismatch!")
        print(f"   Optimization timeframe: {config.timeframe}")
        print(f"   AdvancedLorentzianStrategy timeframe: day (hardcoded)")
        print(f"   ")
        print(f"   🚨 CRITICAL: This mismatch means optimization results")
        print(f"      will NOT translate to real strategy performance!")
        print(f"   ")
        print(f"   🔧 SOLUTION: Set DATA_TIMEFRAME=day in your .env file")
        print(f"      or run: DATA_TIMEFRAME=day python optimize_parameters.py")
        print(f"   ")
        
        response = input(f"   Force daily data for strategy compatibility? (Y/n): ").strip().lower()
        if response in ['n', 'no']:
            print(f"   Exiting...")
            return False
        else:
            print(f"   ✅ Switching to daily data for strategy compatibility")
            config.timeframe = 'day'
            config.aggregate_to_daily = False  # No need to aggregate daily data
    else:
        print(f"✅ Timeframe: {config.timeframe} (matches AdvancedLorentzianStrategy)")
    
    # Data source warning
    print(f"⚠️  Data Source Difference:")
    print(f"   Optimization: Polygon API direct")
    print(f"   AdvancedLorentzianStrategy: Lumibot get_historical_prices()")
    print(f"   ")
    print(f"   📊 Note: Small differences in data may cause minor performance variations")
    print(f"      but the trading logic is now EXACTLY matched.")
    
    # Trading logic confirmation
    print(f"✅ Trading Logic: EXACT match with AdvancedLorentzianStrategy")
    print(f"   • Position sizing: min(cash * 0.95, cash - 1000)")
    print(f"   • start_long: Opens long positions")  
    print(f"   • start_short: Closes long positions (no short selling)")
    print(f"   • Signal processing: Latest signals from classifier")
    print(f"   • Simulation: AdvancedLorentzianSimulator (exact replica)")
    
    print("="*60)
    
    return True

def perform_walk_forward_analysis(param_combinations, df, config):
    """
    Perform walk-forward analysis to test parameter robustness
    This helps avoid overfitting to a single time period
    """
    print(f"\n🔄 WALK-FORWARD ANALYSIS")
    print(f"   Testing top parameters across {config.walk_forward_periods} periods")
    
    # Split data into periods
    total_days = len(df)
    period_size = total_days // config.walk_forward_periods
    
    periods = []
    for i in range(config.walk_forward_periods):
        start_idx = i * period_size
        end_idx = start_idx + period_size if i < config.walk_forward_periods - 1 else total_days
        period_df = df.iloc[start_idx:end_idx]
        periods.append({
            'df': period_df,
            'name': f"Period {i+1}",
            'start_date': period_df.index[0].strftime('%Y-%m-%d'),
            'end_date': period_df.index[-1].strftime('%Y-%m-%d')
        })
    
    # Test top 10 parameter combinations across all periods
    top_params = param_combinations[:10] if len(param_combinations) >= 10 else param_combinations
    
    walk_forward_results = []
    
    for params in tqdm(top_params, desc="Walk-forward testing"):
        period_scores = []
        period_returns = []
        
        for period in periods:
            try:
                result = test_parameter_combination(params, period['df'], config.symbol, config.initial_capital)
                if result:
                    score = calculate_optimization_score(result, config.objectives)
                    period_scores.append(score)
                    period_returns.append(result['total_return'])
                else:
                    period_scores.append(0)
                    period_returns.append(0)
            except:
                period_scores.append(0)
                period_returns.append(0)
        
        # Calculate consistency metrics
        avg_score = np.mean(period_scores)
        std_score = np.std(period_scores)
        consistency_ratio = avg_score / (std_score + 0.001)  # Higher is better
        
        avg_return = np.mean(period_returns)
        std_return = np.std(period_returns)
        
        walk_forward_results.append({
            'parameters': params,
            'avg_score': avg_score,
            'std_score': std_score,
            'consistency_ratio': consistency_ratio,
            'avg_return': avg_return,
            'std_return': std_return,
            'period_scores': period_scores,
            'period_returns': period_returns
        })
    
    # Sort by consistency ratio (most consistent performance)
    walk_forward_results.sort(key=lambda x: x['consistency_ratio'], reverse=True)
    
    print(f"\n🏆 WALK-FORWARD RESULTS (Top 5 Most Consistent):")
    print("="*80)
    
    for i, result in enumerate(walk_forward_results[:5], 1):
        print(f"\n#{i} - Consistency Ratio: {result['consistency_ratio']:.2f}")
        print(f"   Avg Score: {result['avg_score']:.3f} ± {result['std_score']:.3f}")
        print(f"   Avg Return: {result['avg_return']:+.1f}% ± {result['std_return']:.1f}%")
        print(f"   Period Returns: {[f'{r:+.1f}%' for r in result['period_returns']]}")
    
    return walk_forward_results

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
    
    # Show randomization status
    if config.random_seed:
        print(f"   🔒 Randomization: Fixed seed ({config.random_seed}) - reproducible results")
    else:
        print(f"   🎲 Randomization: Random seed - different results each run")
    
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
    
    # Validate optimization vs strategy
    if not validate_optimization_vs_strategy(config):
        return
    
    # Download market data
    print(f"\n📥 Downloading market data...")
    df = download_real_data(symbol=config.symbol, start_date=config.start_date, end_date=config.end_date, timeframe=config.timeframe)
    
    # Handle intraday data aggregation for optimization
    if config.timeframe in ['minute', 'hour'] and config.aggregate_to_daily:
        if is_info():
            print(f"\n📊 Aggregating {config.timeframe} data to daily for optimization...")
        df = aggregate_intraday_to_daily(df, config.timeframe)
        if is_info():
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
        if is_info():
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
            desc = "Testing combinations" if is_info() else "Optimizing"
            with tqdm(total=len(param_combinations), desc=desc, disable=not is_info()) as pbar:
                for i, result in enumerate(pool.imap(test_func, param_combinations)):
                    if result:
                        results.append(result)
                    if is_info():
                        pbar.update(1)
                        pbar.set_postfix({
                            'Valid Results': len(results),
                            'CPU Cores': config.n_jobs,
                            'Success Rate': f"{len(results)/(i+1)*100:.1f}%" if i > 0 else "0%"
                        })
                    
                    # Periodic system monitoring (every 100 combinations)
                    if (i + 1) % 100 == 0 and is_info():
                        cpu_now = psutil.cpu_percent(interval=0.1)
                        if cpu_now > 90:
                            print(f"\n⚠️  High CPU usage detected: {cpu_now:.1f}%")
                            print(f"   Consider stopping (Ctrl+C) and reducing N_JOBS")
    else:
        # Sequential processing for small numbers or when parallel is disabled
        if is_info():
            print(f"🔄 Using sequential processing...")
        desc = "Testing combinations" if is_info() else "Optimizing"
        with tqdm(total=len(param_combinations), desc=desc, disable=not is_info()) as pbar:
            for i, params in enumerate(param_combinations):
                result = test_parameter_combination(params, df_for_classification, config.symbol, config.initial_capital)
                if result:
                    result['optimization_score'] = calculate_optimization_score(result, config.objectives)
                    results.append(result)
                if is_info():
                    pbar.update(1)
                    
                    # Show progress every 10 combinations
                    if (i + 1) % 10 == 0:
                        pbar.set_postfix({'Valid Results': len(results)})
    
    # Show completion summary
    if is_info():
        print(f"\n✅ Optimization completed! Found {len(results)} valid results")
    else:
        print(f"✅ Optimization completed: {len(results)} valid results from {len(param_combinations)} combinations")
    
    # Report error analysis
    failed_combinations = len(param_combinations) - len(results)
    if failed_combinations > 0 and is_info():
        print(f"⚠️  Failed combinations: {failed_combinations} ({failed_combinations/len(param_combinations)*100:.1f}%)")
        if hasattr(test_parameter_combination, 'error_counts'):
            print(f"📊 Error breakdown:")
            for error_type, count in test_parameter_combination.error_counts.items():
                print(f"   • {error_type.replace('_', ' ').title()}: {count}")
    
    if results:
        # Load existing best parameters for comparison
        existing_best = load_existing_best_parameters(config.symbol)
        
        # Display results
        display_optimization_summary(results, config)
        
        # Save results
        save_optimization_results(results, config)
        
        # Save best parameters (with absolute best logic)
        best_params_file = save_best_parameters(results, config, existing_best)
        
        # Show detailed report for the absolute best result (current or existing)
        current_best = max(results, key=lambda x: x['optimization_score'])
        
        if existing_best and 'optimization_info' in existing_best:
            existing_score = existing_best['optimization_info']['optimization_score']
            current_score = current_best['optimization_score']
            
            if current_score > existing_score:
                print(f"\n🏆 DETAILED REPORT FOR NEW BEST PARAMETERS:")
                display_performance_report(current_best)
            else:
                print(f"\n🏆 DETAILED REPORT FOR ABSOLUTE BEST PARAMETERS (Previous Optimization):")
                print(f"📊 Note: Current optimization did not beat the existing best")
                print(f"   Existing best score: {existing_score:.3f} vs Current best: {current_score:.3f}")
                print(f"   Existing best return: {existing_best['optimization_info']['total_return']:+.2f}%")
                print(f"   Found on: {existing_best['optimization_info']['optimization_date'][:10]}")
                print(f"   Total optimization attempts: {existing_best['optimization_info'].get('attempts_count', 1)}")
        else:
            print(f"\n🏆 DETAILED REPORT FOR BEST PARAMETERS:")
            display_performance_report(current_best)
        
        # Perform walk-forward analysis if enabled
        if config.use_walk_forward and len(results) >= 5:
            print(f"\n🔄 STARTING WALK-FORWARD ANALYSIS...")
            # Sort results by score and take top ones for walk-forward testing
            results.sort(key=lambda x: x['optimization_score'], reverse=True)
            top_param_combinations = [r['parameters'] for r in results[:10]]
            walk_forward_results = perform_walk_forward_analysis(top_param_combinations, df_for_classification, config)
        elif config.use_walk_forward:
            print(f"\n⚠️  Skipping walk-forward analysis: need at least 5 valid results, got {len(results)}")
        
    else:
        print("❌ No valid results found. Check your parameter ranges and data.")

if __name__ == "__main__":
    main() 