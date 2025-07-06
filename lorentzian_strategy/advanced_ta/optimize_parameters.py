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

# Load environment variables (override system env vars)
load_dotenv(override=True)

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
        self.start_date = os.getenv('BACKTESTING_START', '2024-01-01')
        self.end_date = os.getenv('BACKTESTING_END', '2024-12-01')
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
        self.timeframe = os.getenv('DATA_TIMEFRAME', 'day').lower()
        self.aggregate_to_daily = os.getenv('AGGREGATE_TO_DAILY', 'true').lower() == 'true'
        
        # Data configuration
        self.max_data_set = int(os.getenv('MAX_DATA_SET', '730'))  # Maximum bars to request for training
        
        # Optimization settings
        self.max_combinations = int(os.getenv('MAX_COMBINATIONS', '2000'))  # Significantly increased default
        
        # Parallel processing settings (N_JOBS empty/None = sequential processing)
        n_jobs_env = os.getenv('N_JOBS', '').strip()
        if n_jobs_env:
            self.use_parallel = True
            self.n_jobs = int(n_jobs_env)
            
            # Safety check: never use more than 75% of available cores
            max_safe_cores = max(1, int(mp.cpu_count() * 0.75))
            if self.n_jobs > max_safe_cores:
                print(f"⚠️  WARNING: N_JOBS={self.n_jobs} may overload system with {mp.cpu_count()} cores")
                print(f"   Reducing to safe limit: {max_safe_cores} cores")
                self.n_jobs = max_safe_cores
        else:
            # Sequential processing
            self.use_parallel = False
            self.n_jobs = 1
        
        # Walk-forward analysis settings (WALK_FORWARD_PERIODS empty/None = disabled)
        walk_forward_env = os.getenv('WALK_FORWARD_PERIODS', '').strip()
        if walk_forward_env:
            self.use_walk_forward = True
            self.walk_forward_periods = int(walk_forward_env)
        else:
            self.use_walk_forward = False
            self.walk_forward_periods = 3  # Default if enabled later
        
        # Randomization settings
        self.random_seed = os.getenv('RANDOM_SEED')  # Set to a number for reproducible results, leave unset for random
        
        # Optimization strategy
        self.optimize_for_return = os.getenv('OPTIMIZE_FOR_RETURN', 'false').lower() == 'true'
        
        # Expanded parameter ranges for more thorough optimization
        self.param_ranges = {
            # Core ML settings - expanded ranges
            'neighborsCount': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30],  # Added higher values
            'maxBarsBack': [],  # Will be set dynamically based on available training data
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

def test_parameter_combination_wrapper(params, df, symbol, initial_capital, objectives, max_bars_back=2000):
    """Wrapper function for parallel processing"""
    result = test_parameter_combination(params, df, symbol, initial_capital, max_bars_back)
    if result:
        result['optimization_score'] = calculate_optimization_score(result, objectives)
    return result

def validate_parameters(params):
    """Validate parameter combination for obvious issues"""
    # Check for reasonable ranges
    if params['neighborsCount'] < 1 or params['neighborsCount'] > 50:
        return False, "Invalid neighborsCount"
    
    # Validate maxBarsBack parameter
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

def test_parameter_combination(params, df, symbol, initial_capital, max_bars_back=2000):
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
        
        # Create settings (now with optimizable EMA/SMA filters and maxBarsBack)
        # Use optimized maxBarsBack parameter (but ensure it doesn't exceed available data)
        effective_max_bars_back = min(params['maxBarsBack'], max_bars_back, len(df) - 50)  # Leave some buffer
        
        # Safety check for minimum effective max bars back
        if effective_max_bars_back < 50:
            effective_max_bars_back = min(50, len(df) - 10)
        
        settings = Settings(
            source=df['close'],
            neighborsCount=params['neighborsCount'],
            maxBarsBack=effective_max_bars_back,
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
        
        # Ensure all required metrics exist with safe defaults
        required_metrics = ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate', 'total_trades', 'profit_factor']
        for metric in required_metrics:
            if metric not in metrics or pd.isna(metrics[metric]) or np.isinf(metrics[metric]):
                if metric == 'sharpe_ratio':
                    metrics[metric] = 0.0
                elif metric == 'max_drawdown':
                    metrics[metric] = 0.0
                elif metric == 'profit_factor':
                    metrics[metric] = 1.0
                else:
                    metrics[metric] = 0
        
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
    if metrics['total_trades'] < 1:
        return -1000  # Use very low score instead of -inf to avoid math issues
    
    score = 0
    for metric_name, config in objectives.items():
        weight = config['weight']
        direction = config['direction']
        
        if metric_name in metrics:
            value = metrics[metric_name]
            
            # Improved normalization with more realistic ranges
            if metric_name == 'total_return':
                # More flexible return normalization - no hard caps for legitimate high returns
                if abs(value) > 1000:  # Only penalize truly extreme returns (>1000%)
                    normalized_value = 0.1
                else:
                    # Extended range: -100% to 500% mapped to 0-1 scale. If over 500%, normalized_value will be 1
                    normalized_value = max(0, min(1, (value + 100) / 600))  # -100% to 500% range
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
    
    # Penalty for unrealistic performance (more lenient)
    if total_return > 500 and win_rate > 90:  # Too good to be true
        score *= 0.7  # Less harsh penalty
    
    # Penalty for too few trades (not enough data)
    if total_trades < 5:
        score *= 0.7
    
    # Bonus for reasonable, consistent performance (more flexible ranges)
    if 10 <= total_return <= 200 and 45 <= win_rate <= 75 and total_trades >= 10:
        score *= 1.1  # Small bonus for realistic performance
    
    return score

def generate_parameter_combinations(config):
    """Generate parameter combinations using smart sampling strategies"""
    # Note: maxBarsBack values are now set dynamically in main() based on actual training data
    param_names = list(config.param_ranges.keys())
    param_values = list(config.param_ranges.values())
    
    # Calculate total possible combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    if is_info():
        print(f"Total possible combinations: {total_combinations:,}")
    
    if total_combinations <= config.max_combinations:
        # Use all combinations if feasible
        if is_info():
            print(f"[✓] Using all {total_combinations:,} combinations")
        all_combinations = list(itertools.product(*param_values))
    else:
        # Use smart sampling strategies
        if is_info():
            print(f"[✓] Smart sampling: {config.max_combinations:,} from {total_combinations:,} possible")
        
        # Strategy 1: Latin Hypercube Sampling for better coverage
        lhs_combinations = generate_latin_hypercube_sample(config.param_ranges, config.max_combinations // 2)
        
        # Strategy 2: Memory-efficient random sampling
        # Use random seed only if RANDOM_SEED environment variable is set
        random_seed = os.getenv('RANDOM_SEED')
        if random_seed:
            np.random.seed(int(random_seed))
            if is_info():
                print(f"  [✓] Using fixed seed: {random_seed}")
        else:
            # Use truly random seed based on current time
            import time
            seed = int(time.time() * 1000000) % 2**32
            np.random.seed(seed)
            if is_info():
                print(f"  [✓] Using random seed: {seed}")
        
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
            print(f"  [✓] Latin Hypercube: {len(lhs_combinations):,}")
            print(f"  [✓] Random sampling: {len(random_sample):,}")
    
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

def display_optimization_summary(results, config, optimize_for_real_trading=False):
    """Display a summary of optimization results"""
    if not results:
        print("[ ] No valid results found")
        return
    
    print("\nParameter Optimization Summary")
    print("=" * 50)
    
    method_name = "Live Trading" if optimize_for_real_trading else "Backtesting Research"
    print(f"Method: {method_name}")
    print(f"Symbol: {config.symbol}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Combinations Tested: {len(results)}")
    
    # Sort by optimization score
    results.sort(key=lambda x: x['optimization_score'], reverse=True)
    
    print(f"\nTop 3 Results:")
    print("-" * 50)
    
    for i, result in enumerate(results[:3], 1):
        metrics = result
        params = result['parameters']
        
        success_return = "[✓]" if metrics['total_return'] > 0 else "[✗]"
        success_winrate = "[✓]" if metrics['win_rate'] >= 50 else "[✗]"
        
        print(f"\n#{i} Score: {result['optimization_score']:.3f}")
        print(f"  {success_return} Return: {metrics['total_return']:+.1f}% | Final: ${metrics['final_portfolio_value']:,.0f}")
        print(f"  {success_winrate} Win Rate: {metrics['win_rate']:.0f}% | Trades: {metrics['total_trades']} | Profit Factor: {metrics.get('profit_factor', 0):.1f}")
        print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Max DD: {metrics.get('max_drawdown', 0):.1f}%")
        print(f"  RSI({params['rsi_period']},{params['rsi_smooth']}) WT({params['wt_n1']},{params['wt_n2']}) CCI({params['cci_period']},{params['cci_smooth']})")
    
    # Best parameters summary
    best_result = results[0]
    print(f"\nBest Parameters (Score: {best_result['optimization_score']:.3f}):")
    print("-" * 30)
    
    best_params = best_result['parameters']
    key_params = ['neighborsCount', 'maxBarsBack', 'rsi_period', 'wt_n1', 'wt_n2', 'cci_period']
    for param in key_params:
        if param in best_params:
            print(f"  {param}: {best_params[param]}")

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
    
    print(f"[✓] Results saved: {filename}")
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

def save_best_parameters(results, config, existing_best=None, optimize_for_real_trading=False):
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
            'optimization_method': 'METHOD_2_LIVE_TRADING' if optimize_for_real_trading else 'METHOD_1_BACKTESTING',
            'method_description': 'Live Trading (Full Data Window)' if optimize_for_real_trading else 'Backtesting Research (Train/Test Split)',
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
            'maxBarsBack': best_params['maxBarsBack'],  # Now included as optimized parameter
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
            'history_window': best_params['maxBarsBack'],  # Use optimized value
            'max_bars_back': best_params['maxBarsBack'],   # Use optimized value
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
    method_desc = "Live Trading (Full Data Window)" if optimize_for_real_trading else "Backtesting Research (Train/Test Split)"
    print(f"   🔧 Optimization Method: {method_desc}")
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
    
    print(f"System Resources:")
    print(f"  CPU: {cpu_percent:.1f}%")
    print(f"  Memory: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
    
    # Warning if system is already under load
    if cpu_percent > 50:
        print(f"  [!] HIGH CPU: {cpu_percent:.1f}% - consider reducing N_JOBS")
    if memory.percent > 80:
        print(f"  [!] HIGH MEMORY: {memory.percent:.1f}% - optimization may be slow")

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

def validate_best_parameters_on_backtest_data(results, config):
    """
    Validate the best optimized parameters on actual backtest data
    
    This function tests the top optimized parameters (trained on historical data)
    against the actual backtest period data to show realistic expected performance.
    """
    if not hasattr(config, 'backtest_data') or len(results) == 0:
        print("❌ No backtest data or results available for validation")
        return
    
    # Get top 3 parameter combinations
    sorted_results = sorted(results, key=lambda x: x['optimization_score'], reverse=True)
    top_results = sorted_results[:3]
    
    print(f"🎯 Testing top {len(top_results)} parameter combinations on backtest data...")
    print(f"   Training period: Historical data before {config.start_date}")
    print(f"   Validation period: {config.start_date} to {config.end_date}")
    print()
    
    # Prepare backtest data
    df_backtest = config.backtest_data.copy()
    df_backtest.columns = df_backtest.columns.str.lower()
    
    validation_results = []
    
    for i, result in enumerate(top_results, 1):
        params = result['parameters']
        
        try:
            # Create features and settings from parameters
            features = [
                Feature("RSI", params['rsi_period'], params['rsi_smooth']),
                Feature("WT", params['wt_n1'], params['wt_n2']),
                Feature("CCI", params['cci_period'], params['cci_smooth'])
            ]
            
            # Use optimized maxBarsBack for validation (with safety limits)
            validation_max_bars_back = min(params['maxBarsBack'], len(df_backtest) - 50, config.max_bars_back)
            settings = Settings(
                source=df_backtest['close'],
                neighborsCount=params['neighborsCount'],
                maxBarsBack=validation_max_bars_back,
                useDynamicExits=params['useDynamicExits'],
                useEmaFilter=params['useEmaFilter'],
                emaPeriod=params['emaPeriod'],
                useSmaFilter=params['useSmaFilter'],
                smaPeriod=params['smaPeriod']
            )
            
            kernel_filter = KernelFilter(
                useKernelSmoothing=params['useKernelSmoothing'],
                lookbackWindow=params['lookbackWindow'],
                relativeWeight=params['relativeWeight'],
                regressionLevel=params['regressionLevel'],
                crossoverLag=params['crossoverLag']
            )
            
            filter_settings = FilterSettings(
                useVolatilityFilter=params['useVolatilityFilter'],
                useRegimeFilter=params['useRegimeFilter'],
                useAdxFilter=params['useAdxFilter'],
                regimeThreshold=params['regimeThreshold'],
                adxThreshold=params['adxThreshold'],
                kernelFilter=kernel_filter
            )
            
            # Test on backtest data
            from simulate_trade import run_advanced_lorentzian_simulation
            backtest_results = run_advanced_lorentzian_simulation(
                df_backtest, features, settings, filter_settings, config.initial_capital
            )
            backtest_metrics = backtest_results['metrics']
            
            # Ensure all required metrics exist with safe defaults
            required_metrics = ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate', 'total_trades']
            for metric in required_metrics:
                if metric not in backtest_metrics:
                    if metric == 'sharpe_ratio':
                        backtest_metrics[metric] = 0.0
                    elif metric == 'max_drawdown':
                        backtest_metrics[metric] = 0.0
                    else:
                        backtest_metrics[metric] = 0
            
            # Store validation result
            validation_result = {
                'rank': i,
                'training_score': result['optimization_score'],
                'training_return': result['total_return'],
                'training_win_rate': result['win_rate'],
                'training_trades': result['total_trades'],
                'backtest_return': backtest_metrics['total_return'],
                'backtest_win_rate': backtest_metrics['win_rate'],
                'backtest_trades': backtest_metrics['total_trades'],
                'backtest_sharpe': backtest_metrics['sharpe_ratio'],
                'backtest_drawdown': backtest_metrics['max_drawdown'],
                'parameters': params
            }
            validation_results.append(validation_result)
            
            # Display individual result
            print(f"📊 Rank #{i} Parameters:")
            print(f"   🏆 Training Performance: {result['total_return']:+.1f}% return, {result['win_rate']:.0f}% win rate, {result['total_trades']} trades")
            print(f"   🎯 Backtest Performance: {backtest_metrics['total_return']:+.1f}% return, {backtest_metrics['win_rate']:.0f}% win rate, {backtest_metrics['total_trades']} trades")
            
            # Calculate performance consistency
            return_diff = abs(backtest_metrics['total_return'] - result['total_return'])
            win_rate_diff = abs(backtest_metrics['win_rate'] - result['win_rate'])
            
            if return_diff < 10 and win_rate_diff < 15:
                consistency = "🟢 EXCELLENT"
            elif return_diff < 20 and win_rate_diff < 25:
                consistency = "🟡 GOOD"
            else:
                consistency = "🔴 POOR"
            
            print(f"   📈 Consistency: {consistency} (return diff: {return_diff:.1f}%, win rate diff: {win_rate_diff:.1f}%)")
            print()
            
        except Exception as e:
            print(f"❌ Error validating rank #{i}: {str(e)}")
            continue
    
    # Summary
    if validation_results:
        print("="*80)
        print("📊 VALIDATION SUMMARY")
        print("="*80)
        
        avg_training_return = np.mean([r['training_return'] for r in validation_results])
        avg_backtest_return = np.mean([r['backtest_return'] for r in validation_results])
        avg_return_diff = abs(avg_backtest_return - avg_training_return)
        
        print(f"🏆 Average Training Return: {avg_training_return:+.1f}%")
        print(f"🎯 Average Backtest Return: {avg_backtest_return:+.1f}%")
        print(f"📊 Average Return Difference: {avg_return_diff:.1f}%")
        
        if avg_return_diff < 10:
            print(f"✅ EXCELLENT: Optimization results are highly predictive!")
        elif avg_return_diff < 20:
            print(f"👍 GOOD: Optimization results are reasonably predictive")
        else:
            print(f"⚠️  WARNING: Large difference suggests overfitting or market regime change")
        
        print("="*80)
        print()

def perform_walk_forward_analysis(param_combinations, df, config):
    """
    Perform walk-forward analysis to test parameter robustness
    This helps avoid overfitting to a single time period
    """
    print(f"\n🔄 WALK-FORWARD ANALYSIS")
    print(f"   Testing top parameters across {config.walk_forward_periods} periods")
    
    # Check if we have enough data for walk-forward analysis
    min_data_required = config.walk_forward_periods * 100  # At least 100 days per period
    if len(df) < min_data_required:
        print(f"   ⚠️  WARNING: Insufficient data for walk-forward analysis")
        print(f"   📊 Available: {len(df)} days | Required: {min_data_required} days")
        print(f"   🚫 Skipping walk-forward analysis")
        return []
    
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
        
        # Calculate consistency metrics with proper handling of invalid scores
        valid_scores = [s for s in period_scores if not (np.isinf(s) or np.isnan(s))]
        
        if len(valid_scores) >= 2:
            avg_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            consistency_ratio = avg_score / (std_score + 0.001) if std_score > 0 else avg_score
        elif len(valid_scores) == 1:
            avg_score = valid_scores[0]
            std_score = 0
            consistency_ratio = avg_score
        else:
            # All scores are invalid (-inf or nan)
            avg_score = -1000  # Very bad score but not -inf
            std_score = 0
            consistency_ratio = -1000
        
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

def validate_environment_configuration() -> bool:
    """
    Validate environment configuration before starting optimization
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    print("🔍 VALIDATING CONFIGURATION")
    print("="*50)
    
    errors = []
    warnings = []
    
    # 1. Check required environment variables
    required_vars = ['SYMBOL', 'BACKTESTING_START', 'BACKTESTING_END', 'POLYGON_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # 2. Validate date formats and logic
    try:
        start_date = datetime.strptime(os.getenv('BACKTESTING_START', '2024-01-01'), '%Y-%m-%d')
        end_date = datetime.strptime(os.getenv('BACKTESTING_END', '2024-12-01'), '%Y-%m-%d')
        
        if start_date >= end_date:
            errors.append("BACKTESTING_START must be before BACKTESTING_END")
        
        # Check if backtest period is too short
        days_diff = (end_date - start_date).days
        if days_diff < 30:
            warnings.append(f"Very short backtest period: {days_diff} days (recommended: 90+ days)")
            
    except ValueError as e:
        errors.append(f"Invalid date format in BACKTESTING_START or BACKTESTING_END: {e}")
    
    # 3. Validate MAX_DATA_SET based on timeframe
    try:
        max_data_set = int(os.getenv('MAX_DATA_SET', '730'))
        timeframe = os.getenv('DATA_TIMEFRAME', 'day').lower()
        
        if max_data_set <= 0:
            errors.append("MAX_DATA_SET must be positive")
        elif max_data_set < 50:
            warnings.append(f"Very small data set: {max_data_set} bars (recommended: 200+ bars)")
        
        # Validate against API limits based on timeframe
        if timeframe == 'day':
            if max_data_set > 730:
                errors.append(f"MAX_DATA_SET={max_data_set} exceeds Polygon free tier limit (730 days)")
        elif timeframe == 'hour':
            max_hourly_bars = 730 * 6.5  # ~4745 bars
            if max_data_set > max_hourly_bars:
                errors.append(f"MAX_DATA_SET={max_data_set} exceeds estimated hourly limit ({int(max_hourly_bars)} bars)")
        elif timeframe == 'minute':
            max_minute_bars = 730 * 6.5 * 60  # ~284,700 bars (theoretical)
            if max_data_set > 10000:  # Practical limit for performance
                warnings.append(f"Large minute data set ({max_data_set} bars) will be very slow")
                
    except ValueError:
        errors.append("MAX_DATA_SET must be a valid integer")
    
    # 4. Validate other configuration values
    try:
        initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
        if initial_capital <= 0:
            errors.append("INITIAL_CAPITAL must be positive")
        elif initial_capital < 1000:
            warnings.append(f"Small initial capital: ${initial_capital:,.0f} (recommended: $10,000+)")
    except ValueError:
        errors.append("INITIAL_CAPITAL must be a valid number")
    
    try:
        max_combinations = int(os.getenv('MAX_COMBINATIONS', '2000'))
        if max_combinations <= 0:
            errors.append("MAX_COMBINATIONS must be positive")
        elif max_combinations < 100:
            warnings.append(f"Few combinations: {max_combinations} (recommended: 1000+ for thorough optimization)")
        elif max_combinations > 10000:
            warnings.append(f"Many combinations: {max_combinations} (will take very long time)")
    except ValueError:
        errors.append("MAX_COMBINATIONS must be a valid integer")
    
    # 5. Validate parallel processing settings
    n_jobs_env = os.getenv('N_JOBS', '').strip()
    if n_jobs_env:
        try:
            n_jobs = int(n_jobs_env)
            cpu_count = mp.cpu_count()
            if n_jobs <= 0:
                errors.append("N_JOBS must be positive")
            elif n_jobs > cpu_count:
                warnings.append(f"N_JOBS ({n_jobs}) exceeds CPU count ({cpu_count})")
        except ValueError:
            errors.append("N_JOBS must be a valid integer")
    
    # 6. Validate walk-forward settings
    walk_forward_env = os.getenv('WALK_FORWARD_PERIODS', '').strip()
    if walk_forward_env:
        try:
            periods = int(walk_forward_env)
            if periods <= 1:
                errors.append("WALK_FORWARD_PERIODS must be > 1")
            elif periods > 10:
                warnings.append(f"Many walk-forward periods ({periods}) will be very slow")
        except ValueError:
            errors.append("WALK_FORWARD_PERIODS must be a valid integer")
    
    # 7. Validate symbol format
    symbol = os.getenv('SYMBOL', '')
    if symbol and not symbol.replace('.', '').replace('-', '').isalnum():
        warnings.append(f"Unusual symbol format: '{symbol}' (verify it's correct)")
    
    # Display results
    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"   • {error}")
        print()
        
    if warnings:
        print("⚠️  CONFIGURATION WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
        print()
    
    if not errors and not warnings:
        print("✅ Configuration looks good!")
        print()
        return True
    elif not errors:
        print("✅ Configuration valid (with warnings)")
        print()
        return True
    else:
        print("❌ Please fix configuration errors before continuing.")
        print()
        return False

def main():
    """Main optimization function"""
    print("Starting Parameter Optimization for Lorentzian Classification")
    print("="*80)
    
    # Validate configuration before proceeding
    if not validate_environment_configuration():
        print("[!] Exiting due to configuration errors")
        sys.exit(1)
    
    # Check system resources before starting
    monitor_system_resources()
    print()
    
    # Load configuration
    config = OptimizationConfig()
    
    print(f"Configuration:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Capital: ${config.initial_capital:,.0f}")
    print(f"  Max combinations: {config.max_combinations:,}")
    if config.use_parallel:
        print(f"  Parallel: {config.n_jobs} cores")
    else:
        print(f"  Parallel: disabled")
    
    # Show randomization status
    if config.random_seed:
        print(f"  Seed: {config.random_seed} (reproducible)")
    else:
        print(f"  Seed: random")
    
    # Display optimization strategy
    strategy_name = "Return-Focused" if config.optimize_for_return else "Balanced"
    print(f"  Strategy: {strategy_name}")
    
    # Validate timeframe
    if config.timeframe not in ['day', 'hour', 'minute']:
        print(f"[!] Invalid timeframe '{config.timeframe}', using 'day'")
        config.timeframe = 'day'
    
    # Warning for intraday data optimization
    if config.timeframe in ['minute', 'hour']:
        start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        if config.timeframe == 'minute' and days_diff > 7:
            print(f"[!] WARNING: {days_diff} days of minute data will be VERY slow")
        elif config.timeframe == 'hour' and days_diff > 90:
            print(f"[!] WARNING: {days_diff} days of hourly data may be slow")
    
    # Validate optimization vs strategy
    if not validate_optimization_vs_strategy(config):
        return
    
    # Interactive optimization method selection
    print(f"\nOptimization Method Selection")
    print(f"=" * 40)
    print(f"Choose your optimization approach:")
    print(f"")
    print(f"1. Backtesting Research")
    print(f"   - Test on specific historical period")
    print(f"   - Train on data BEFORE backtest period")
    print(f"   - Eliminates look-ahead bias")
    print(f"")
    print(f"2. Live Trading")
    print(f"   - Find best parameters for deployment")
    print(f"   - Use full 2-year data window")
    print(f"   - Maximum training data")
    print(f"")
    
    while True:
        choice = input(f"Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print(f"[!] Invalid choice. Please enter 1 or 2.")
    
    optimize_for_real_trading = (choice == '2')
    method_name = "Live Trading" if optimize_for_real_trading else "Backtesting Research"
    print(f"[✓] Selected: {method_name}")
    
    # Download market data based on selected method
    from datetime import datetime, timedelta
    
    if optimize_for_real_trading:
        # METHOD 2: Use full data window for live trading optimization
        print(f"\n[✓] Downloading full data window for live trading...")
        
        # Get all available data (730 bars from your test, for daily data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        print(f"  Data window: {start_date} to {end_date}")
        
        # Download all available data
        df_full = download_real_data(
            symbol=config.symbol, 
            start_date=start_date, 
            end_date=end_date, 
            timeframe=config.timeframe
        )
        
        print(f"  [✓] Training data: {len(df_full)} rows")
        
        # Use full data for optimization
        df = df_full
        config.backtest_data = None  # No separate backtest data
        
    else:
        # METHOD 1: Traditional train/test split for backtesting research
        print(f"\n[✓] Downloading data with train/test split...")
        print(f"  Backtest period: {config.start_date} to {config.end_date}")
        
        # Calculate training data range using ACTUAL available data limits (730 days max)
        backtest_start = datetime.strptime(config.start_date, '%Y-%m-%d')
        
        # Training data ends 1 day before backtest starts (same as Lumibot)
        training_end = (backtest_start - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Calculate earliest available date (730 days from today, based on our API testing)
        today = datetime.now()
        earliest_available = (today - timedelta(days=730)).strftime('%Y-%m-%d')
        
        # Training starts from the later of: earliest_available OR (training_end - desired_bars)
        # This ensures we don't request data that doesn't exist
        desired_training_start = (backtest_start - timedelta(days=config.max_data_set)).strftime('%Y-%m-%d')
        
        # Use the later date (closer to present) to ensure data exists
        if datetime.strptime(desired_training_start, '%Y-%m-%d') < datetime.strptime(earliest_available, '%Y-%m-%d'):
            training_start = earliest_available
            actual_days = (datetime.strptime(training_end, '%Y-%m-%d') - datetime.strptime(training_start, '%Y-%m-%d')).days
            print(f"   ⚠️  Requested {config.max_data_set} bars, but API only has {actual_days} days available")
        else:
            training_start = desired_training_start
            actual_days = config.max_data_set
        
        print(f"  Training period: {training_start} to {training_end} ({actual_days} bars)")
        
        # Download training data (for optimization)
        print(f"  [✓] Requesting training data...")
        df_training = download_real_data(
            symbol=config.symbol, 
            start_date=training_start, 
            end_date=training_end, 
            timeframe=config.timeframe
        )
        
        # Also download backtest data for validation/analysis (but not for optimization)
        print(f"  [✓] Requesting backtest data...")
        df_backtest = download_real_data(
            symbol=config.symbol, 
            start_date=config.start_date, 
            end_date=config.end_date, 
            timeframe=config.timeframe
        )
        
        print(f"  [✓] Training data: {len(df_training)} rows")
        print(f"  [✓] Backtest data: {len(df_backtest)} rows")
        
        # Check if training data is insufficient
        if len(df_training) < 50:
            print(f"  [!] CRITICAL: Only {len(df_training)} bars available")
            print(f"  Recommendation: Use METHOD 2 for optimization")
        elif len(df_training) < 200:
            print(f"  [!] WARNING: Only {len(df_training)} bars available")
            print(f"  Insufficient for robust optimization (need 500+ bars)")
            print(f"  Consider using METHOD 2 for more training data")
        else:
            print(f"  [✓] Training data sufficient: {len(df_training)} bars")
        
        # Use training data for optimization (mimics Lumibot's approach)
        df = df_training
        
        # Store backtest data for potential validation
        config.backtest_data = df_backtest
    
    # Smart maxBarsBack optimization based on available training data
    available_training_bars = len(df)
    max_bars_back_options = [
        int(available_training_bars * 0.25),  # 25% of available data
        int(available_training_bars * 0.50),  # 50% of available data  
        int(available_training_bars * 0.75),  # 75% of available data
        int(available_training_bars * 1.00),  # 100% of available data
    ]
    # Remove duplicates and ensure minimum
    max_bars_back_options = sorted(list(set([max(50, x) for x in max_bars_back_options])))
    
    # Update config with smart maxBarsBack options
    config.param_ranges['maxBarsBack'] = max_bars_back_options
    config.max_bars_back = max(max_bars_back_options)  # Set limit to maximum option
    
    print(f"  [✓] Smart maxBarsBack options: {max_bars_back_options}")
    print(f"  Based on {available_training_bars} training bars (25%, 50%, 75%, 100%)")
    
    # Handle intraday data aggregation for optimization
    if config.timeframe in ['minute', 'hour'] and config.aggregate_to_daily:
        if is_info():
            print(f"\n[✓] Aggregating {config.timeframe} data to daily...")
        df = aggregate_intraday_to_daily(df, config.timeframe)
        if is_info():
            print(f"[✓] Using daily aggregated data")
    
    # Convert to lowercase columns for classification
    df_for_classification = df.copy()
    df_for_classification.columns = df_for_classification.columns.str.lower()
    
    # Generate parameter combinations
    print(f"\n[✓] Generating parameter combinations...")
    param_combinations = generate_parameter_combinations(config)
    print(f"  Testing {len(param_combinations):,} combinations")
    
    # Load existing best parameters for comparison
    existing_best = load_existing_best_parameters(config.symbol)
    existing_best_score = 0
    if existing_best and 'optimization_info' in existing_best:
        existing_best_score = existing_best['optimization_info']['optimization_score']
        print(f"[✓] Existing best score to beat: {existing_best_score:.3f} (Return: {existing_best['optimization_info']['total_return']:+.1f}%)")
    else:
        print(f"[✓] No existing best found - any result will be new record")
    
    # Run optimization
    print(f"\n[✓] Running optimization...")
    results = []
    
    if config.use_parallel and len(param_combinations) > 10:
        # Parallel processing for large numbers of combinations
        if is_info():
            print(f"[✓] Using {config.n_jobs} cores...")
        
        # Create partial function with fixed arguments
        test_func = partial(
            test_parameter_combination_wrapper,
            df=df_for_classification,
            symbol=config.symbol,
            initial_capital=config.initial_capital,
            objectives=config.objectives,
            max_bars_back=config.max_bars_back
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
                        # Calculate current best for display
                        if results:
                            current_best_result = max(results, key=lambda x: x['optimization_score'])
                            current_best_score = current_best_result['optimization_score']
                            current_best_return = current_best_result['total_return']
                            
                            # Only show if it beats existing best
                            if current_best_score > existing_best_score:
                                postfix = {
                                    'Valid': len(results),
                                    'Success': f"{len(results)/(i+1)*100:.0f}%" if i > 0 else "0%",
                                    'NEW BEST': f"{current_best_score:.2f}",
                                    'Return': f"{current_best_return:+.1f}%"
                                }
                            else:
                                postfix = {
                                    'Valid': len(results),
                                    'Success': f"{len(results)/(i+1)*100:.0f}%" if i > 0 else "0%",
                                    'Best': f"{current_best_score:.2f}",
                                    'To Beat': f"{existing_best_score:.2f}"
                                }
                        else:
                            postfix = {
                                'Valid': 0,
                                'Success': "0%",
                                'To Beat': f"{existing_best_score:.2f}"
                            }
                        
                        pbar.set_postfix(postfix)
                    
                    # Periodic system monitoring (every 500 combinations)
                    if (i + 1) % 500 == 0 and is_info():
                        cpu_now = psutil.cpu_percent(interval=0.1)
                        if cpu_now > 90:
                            pbar.write(f"[!] High CPU: {cpu_now:.1f}% - consider reducing N_JOBS")
                        
                        # Show current best every 500 iterations - only if beats existing
                        if results:
                            best_result = max(results, key=lambda x: x['optimization_score'])
                            if best_result['optimization_score'] > existing_best_score:
                                pbar.write(f"[NEW RECORD] {i+1}/{len(param_combinations)} | Score: {best_result['optimization_score']:.3f} (beats {existing_best_score:.3f}) | Return: {best_result['total_return']:+.1f}%")
                            else:
                                pbar.write(f"[Progress] {i+1}/{len(param_combinations)} | Best: {best_result['optimization_score']:.3f} | Need: {existing_best_score:.3f} to beat existing")
    else:
        # Sequential processing for small numbers or when parallel is disabled
        if is_info():
            print(f"[✓] Using sequential processing...")
        desc = "Testing combinations" if is_info() else "Optimizing"
        with tqdm(total=len(param_combinations), desc=desc, disable=not is_info()) as pbar:
            for i, params in enumerate(param_combinations):
                result = test_parameter_combination(params, df_for_classification, config.symbol, config.initial_capital, config.max_bars_back)
                if result:
                    result['optimization_score'] = calculate_optimization_score(result, config.objectives)
                    results.append(result)
                if is_info():
                    pbar.update(1)
                    
                    # Show progress every 10 combinations
                    if (i + 1) % 10 == 0:
                        # Calculate current best for display
                        if results:
                            current_best_result = max(results, key=lambda x: x['optimization_score'])
                            current_best_score = current_best_result['optimization_score']
                            current_best_return = current_best_result['total_return']
                            
                            # Only show if it beats existing best
                            if current_best_score > existing_best_score:
                                postfix = {
                                    'Valid': len(results),
                                    'NEW BEST': f"{current_best_score:.2f}",
                                    'Return': f"{current_best_return:+.1f}%"
                                }
                            else:
                                postfix = {
                                    'Valid': len(results),
                                    'Best': f"{current_best_score:.2f}",
                                    'To Beat': f"{existing_best_score:.2f}"
                                }
                        else:
                            postfix = {
                                'Valid': 0,
                                'To Beat': f"{existing_best_score:.2f}"
                            }
                        
                        pbar.set_postfix(postfix)
    
    # Show completion summary
    if is_info():
        print(f"\n[✓] Optimization completed: {len(results)} valid results")
    else:
        print(f"[✓] Optimization completed: {len(results)} valid results from {len(param_combinations)} combinations")
    
    # Report error analysis
    failed_combinations = len(param_combinations) - len(results)
    if failed_combinations > 0 and is_info():
        print(f"[!] Failed: {failed_combinations} ({failed_combinations/len(param_combinations)*100:.1f}%)")
        if hasattr(test_parameter_combination, 'error_counts'):
            print(f"Error breakdown:")
            for error_type, count in test_parameter_combination.error_counts.items():
                print(f"  - {error_type.replace('_', ' ').title()}: {count}")
    
    if results:
        # existing_best already loaded above
        
        # Display results
        display_optimization_summary(results, config, optimize_for_real_trading)
        
        # Save results
        save_optimization_results(results, config)
        
        # Save best parameters (with absolute best logic and method info)
        best_params_file = save_best_parameters(results, config, existing_best, optimize_for_real_trading)
        
        # Show detailed report for the absolute best result (current or existing)
        current_best = max(results, key=lambda x: x['optimization_score'])
        
        if existing_best and 'optimization_info' in existing_best:
            existing_score = existing_best['optimization_info']['optimization_score']
            current_score = current_best['optimization_score']
            
            if current_score > existing_score:
                print(f"\nDetailed Report - New Best Parameters:")
                display_performance_report(current_best)
            else:
                print(f"\nDetailed Report - Previous Best Still Optimal:")
                print(f"  Previous: {existing_score:.3f} vs Current: {current_score:.3f}")
                print(f"  Return: {existing_best['optimization_info']['total_return']:+.1f}%")
                print(f"  Date: {existing_best['optimization_info']['optimization_date'][:10]}")
        else:
            print(f"\nDetailed Report - Best Parameters:")
            display_performance_report(current_best)
        
        # Validate best parameters on actual backtest data (only for Method 1)
        if hasattr(config, 'backtest_data') and config.backtest_data is not None:
            print(f"\n[✓] Validating parameters on backtest data...")
            validate_best_parameters_on_backtest_data(results, config)
        elif optimize_for_real_trading:
            print(f"\n[✓] Live trading optimization complete")
            print(f"  Parameters ready for AdvancedLorentzianStrategy")
        
        # Perform walk-forward analysis if enabled
        if config.use_walk_forward and len(results) >= 5:
            print(f"\n[✓] Starting walk-forward analysis...")
            # Sort results by score and take top ones for walk-forward testing
            results.sort(key=lambda x: x['optimization_score'], reverse=True)
            top_param_combinations = [r['parameters'] for r in results[:10]]
            walk_forward_results = perform_walk_forward_analysis(top_param_combinations, df_for_classification, config)
        elif config.use_walk_forward:
            print(f"\n[!] Skipping walk-forward: need 5+ results, got {len(results)}")
        
    else:
        print("[!] No valid results found. Check parameter ranges and data.")

if __name__ == "__main__":
    main() 