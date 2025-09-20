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

# Configure matplotlib backend FIRST before any other imports
import os
os.environ['MPLBACKEND'] = 'Agg'  # Set non-interactive backend via environment

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import itertools
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import psutil  # For system monitoring
import pickle
import hashlib
from collections import deque
import logging

# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# Suppress GUI-related warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*backend.*')

# Load environment variables (override system env vars)
load_dotenv(override=True)

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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


class SmartOptimizer:
    """
    Smart optimization system that continues from where it left off
    and uses intelligent search strategies
    """
    
    def __init__(self, config):
        self.config = config
        self.symbol = config.symbol
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
        self.state_file = os.path.join(self.results_dir, f"optimization_state_{self.symbol}.pkl")
        self.tested_combinations = set()
        self.top_performers = deque(maxlen=100)  # Keep top 100 results
        self.generation = 0
        self.best_score = -float('inf')
        self.best_params = None
        self.stagnation_counter = 0
        self.max_stagnation = 50  # Restart strategy after 50 generations without improvement
        
    def save_state(self):
        """Save optimization state to resume later"""
        if not self.config.save_optimization_state:
            return
            
        os.makedirs(self.results_dir, exist_ok=True)
        state = {
            'tested_combinations': self.tested_combinations,
            'top_performers': list(self.top_performers),
            'generation': self.generation,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'stagnation_counter': self.stagnation_counter,
            'param_ranges': self.config.param_ranges
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Load previous optimization state"""
        if not self.config.continue_from_best or not os.path.exists(self.state_file):
            return False
            
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            
            self.tested_combinations = state.get('tested_combinations', set())
            self.top_performers = deque(state.get('top_performers', []), maxlen=100)
            self.generation = state.get('generation', 0)
            self.best_score = state.get('best_score', -float('inf'))
            self.best_params = state.get('best_params', None)
            self.stagnation_counter = state.get('stagnation_counter', 0)
            
            logger.info(f"Loaded optimization state: Generation {self.generation}, "
                       f"Tested combinations: {len(self.tested_combinations):,}, "
                       f"Best score: {self.best_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load optimization state: {e}")
            return False
    
    def params_to_hash(self, params):
        """Convert parameters to hashable string for tracking"""
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def is_combination_tested(self, params):
        """Check if parameter combination was already tested"""
        return self.params_to_hash(params) in self.tested_combinations
    
    def mark_combination_tested(self, params, result):
        """Mark parameter combination as tested and store result"""
        param_hash = self.params_to_hash(params)
        self.tested_combinations.add(param_hash)
        
        if result and 'optimization_score' in result:
            self.top_performers.append(result)
            
            # Update best if this is better
            if result['optimization_score'] > self.best_score:
                self.best_score = result['optimization_score']
                self.best_params = params.copy()
                self.stagnation_counter = 0
                logger.info(f"NEW BEST - Generation {self.generation}: Score {self.best_score:.3f}, "
                           f"Return {result['total_return']:+.1f}%")
            else:
                self.stagnation_counter += 1
    
    def generate_smart_combinations(self, n_combinations):
        """Generate smart parameter combinations using multiple strategies"""
        combinations = []
        self.generation += 1
        
        # Strategy allocation based on performance and exploration needs
        if len(self.top_performers) < 10:
            # Early exploration phase
            random_ratio = 0.8
            local_ratio = 0.15
            genetic_ratio = 0.05
        elif self.stagnation_counter > 20:
            # Stuck in local optimum - increase exploration
            random_ratio = 0.5
            local_ratio = 0.3
            genetic_ratio = 0.2
        else:
            # Normal exploitation phase
            random_ratio = 0.3
            local_ratio = 0.5
            genetic_ratio = 0.2
        
        # Generate combinations using different strategies
        n_random = int(n_combinations * random_ratio)
        n_local = int(n_combinations * local_ratio)
        n_genetic = n_combinations - n_random - n_local
        
        combinations.extend(self._generate_random_combinations(n_random))
        combinations.extend(self._generate_local_search_combinations(n_local))
        combinations.extend(self._generate_genetic_combinations(n_genetic))
        
        return combinations
    
    def _generate_random_combinations(self, n):
        """Generate random parameter combinations"""
        combinations = []
        for _ in range(n):
            params = {}
            for param, values in self.config.param_ranges.items():
                if values:  # Skip empty parameter ranges
                    params[param] = np.random.choice(values)
            combinations.append(params)
        return combinations
    
    def _generate_local_search_combinations(self, n):
        """Generate combinations around top performers"""
        if not self.top_performers:
            return self._generate_random_combinations(n)
        
        combinations = []
        for _ in range(n):
            # Select a random top performer
            base_result = np.random.choice(self.top_performers)
            base_params = base_result['parameters']
            
            # Create variation
            new_params = self._create_parameter_variation(base_params)
            combinations.append(new_params)
        
        return combinations
    
    def _generate_genetic_combinations(self, n):
        """Generate combinations using genetic algorithm principles"""
        if len(self.top_performers) < 2:
            return self._generate_random_combinations(n)
        
        combinations = []
        for _ in range(n):
            # Select two parents from top performers
            parent1 = np.random.choice(self.top_performers)['parameters']
            parent2 = np.random.choice(self.top_performers)['parameters']
            
            # Create child through crossover
            child = self._crossover_parameters(parent1, parent2)
            
            # Apply mutation
            child = self._mutate_parameters(child)
            
            combinations.append(child)
        
        return combinations
    
    def _create_parameter_variation(self, base_params):
        """Create a variation of base parameters"""
        new_params = base_params.copy()
        
        # Randomly modify 1-3 parameters
        num_changes = np.random.randint(1, min(4, len(base_params) + 1))
        params_to_change = np.random.choice(list(base_params.keys()), num_changes, replace=False)
        
        for param in params_to_change:
            if param in self.config.param_ranges and self.config.param_ranges[param]:
                new_params[param] = np.random.choice(self.config.param_ranges[param])
        
        return new_params
    
    def _crossover_parameters(self, parent1, parent2):
        """Create child parameters through crossover"""
        child = {}
        
        for param in parent1.keys():
            if param in parent2:
                # Randomly choose from either parent
                child[param] = np.random.choice([parent1[param], parent2[param]])
            else:
                child[param] = parent1[param]
        
        return child
    
    def _mutate_parameters(self, params):
        """Apply mutation to parameters"""
        mutated = params.copy()
        
        # Mutate with probability
        mutation_rate = 0.1
        for param, value in mutated.items():
            if np.random.random() < mutation_rate:
                if param in self.config.param_ranges and self.config.param_ranges[param]:
                    mutated[param] = np.random.choice(self.config.param_ranges[param])
        
        return mutated


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
        
        # Parallel processing settings
        self._setup_parallel_processing()
        
        # Walk-forward analysis settings
        self._setup_walk_forward()
        
        # Randomization settings
        self.random_seed = os.getenv('RANDOM_SEED')
        
        # Smart optimization settings
        self.use_smart_optimization = os.getenv('USE_SMART_OPTIMIZATION', 'true').lower() == 'true'
        self.smart_exploration_ratio = float(os.getenv('SMART_EXPLORATION_RATIO', '0.3'))
        self.adaptive_search_radius = float(os.getenv('ADAPTIVE_SEARCH_RADIUS', '0.2'))
        self.continue_from_best = os.getenv('CONTINUE_FROM_BEST', 'true').lower() == 'true'
        self.save_optimization_state = os.getenv('SAVE_OPTIMIZATION_STATE', 'true').lower() == 'true'
        
        # Optimization strategy
        self.optimize_for_return = os.getenv('OPTIMIZE_FOR_RETURN', 'false').lower() == 'true'
        
        # Advanced optimization methods
        self.use_bayesian_optimization = os.getenv('USE_BAYESIAN_OPTIMIZATION', 'false').lower() == 'true'
        self.use_genetic_algorithm = os.getenv('USE_GENETIC_ALGORITHM', 'false').lower() == 'true'
        
        # Set parameter ranges
        self._setup_parameter_ranges()
    
    def _setup_parallel_processing(self):
        """Setup parallel processing configuration"""
        n_jobs_env = os.getenv('N_JOBS', '').strip()
        if n_jobs_env:
            self.use_parallel = True
            self.n_jobs = int(n_jobs_env)
            
            # Safety check: never use more than 75% of available cores
            max_safe_cores = max(1, int(mp.cpu_count() * 0.75))
            if self.n_jobs > max_safe_cores:
                logger.warning(f"N_JOBS={self.n_jobs} may overload system with {mp.cpu_count()} cores. "
                              f"Reducing to safe limit: {max_safe_cores} cores")
                self.n_jobs = max_safe_cores
        else:
            self.use_parallel = False
            self.n_jobs = 1
    
    def _setup_walk_forward(self):
        """Setup walk-forward analysis configuration"""
        walk_forward_env = os.getenv('WALK_FORWARD_PERIODS', '').strip()
        if walk_forward_env:
            self.use_walk_forward = True
            self.walk_forward_periods = int(walk_forward_env)
        else:
            self.use_walk_forward = False
            self.walk_forward_periods = 3
    
    def _setup_parameter_ranges(self):
        """Setup parameter ranges for optimization"""
        self.param_ranges = {
            # Core ML settings
            'neighborsCount': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30],
            'maxBarsBack': [],  # Will be set dynamically
            'useDynamicExits': [True, False],
            
            # RSI Feature parameters
            'rsi_period': [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40],
            'rsi_smooth': [1, 2, 3, 4, 5, 6, 7, 8],
            
            # Williams %R (WT) Feature parameters
            'wt_n1': [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20],
            'wt_n2': [4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 22, 25],
            
            # CCI Feature parameters
            'cci_period': [6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30],
            'cci_smooth': [1, 2, 3, 4, 5, 6, 7, 8],
            
            # EMA/SMA Filter settings
            'useEmaFilter': [True, False],
            'emaPeriod': [20, 30, 50, 75, 100, 150, 200, 250, 300],
            'useSmaFilter': [True, False],
            'smaPeriod': [20, 30, 50, 75, 100, 150, 200, 250, 300],
            
            # Advanced filter settings
            'useVolatilityFilter': [True, False],
            'useRegimeFilter': [True, False],
            'useAdxFilter': [True, False],
            'regimeThreshold': [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            'adxThreshold': [10, 15, 20, 25, 30, 35, 40, 45],
            
            # Kernel filter settings
            'useKernelSmoothing': [True, False],
            'lookbackWindow': [2, 4, 6, 8, 10, 12, 14, 16, 20, 25],
            'relativeWeight': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
            'regressionLevel': [10, 15, 20, 25, 30, 35, 40, 50],
            'crossoverLag': [1, 2, 3, 4, 5, 6, 8, 10],
        }


# Utility Functions
def get_next_file_number(directory: str, pattern: str) -> int:
    """Get the next available file number for incremental naming"""
    if not os.path.exists(directory):
        return 1
    
    files = os.listdir(directory)
    max_num = 0
    
    for file in files:
        if pattern in file:
            try:
                # Extract number from filename
                parts = file.split('_')
                for part in parts:
                    if part.isdigit():
                        max_num = max(max_num, int(part))
            except:
                continue
    
    return max_num + 1


def generate_incremental_filename(directory: str, base_filename: str, extension: str) -> str:
    """Generate incremental filename to avoid overwrites"""
    next_num = get_next_file_number(directory, base_filename)
    return f"{base_filename}_{next_num:03d}.{extension}"


def validate_parameters(params):
    """Validate parameter values are within acceptable ranges"""
    try:
        # Basic type and range validation
        if params.get('neighborsCount', 0) < 1:
            return False, "neighborsCount must be >= 1"
        
        if params.get('maxBarsBack', 0) < 50:
            return False, "maxBarsBack must be >= 50"
        
        # RSI validation
        if not (1 <= params.get('rsi_period', 14) <= 100):
            return False, "rsi_period must be between 1 and 100"
        
        # Additional validations can be added here
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def test_parameter_combination_wrapper(params, df, symbol, initial_capital, objectives, max_bars_back=2000):
    """Wrapper function for parallel processing"""
    return test_parameter_combination(params, df, symbol, initial_capital, max_bars_back)


def test_parameter_combination(params, df, symbol, initial_capital, max_bars_back=2000):
    """Test a specific parameter combination and return performance metrics"""
    try:
        # Validate parameters first
        is_valid, validation_msg = validate_parameters(params)
        if not is_valid:
            logger.debug(f"Invalid parameters: {validation_msg}")
            return None
        
        # Set maxBarsBack dynamically if not provided
        if 'maxBarsBack' not in params or params['maxBarsBack'] == 0:
            params['maxBarsBack'] = min(max_bars_back, len(df) - 50)
        
        # Import and run classifier
        from classifier import LorentzianClassifier
        
        classifier = LorentzianClassifier(
            df=df,
            symbol=symbol,
            **params
        )
        
        # Run classification
        results = classifier.run_classification()
        
        if results is None or results.empty:
            logger.debug("Classification returned empty results")
            return None
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(results, initial_capital)
        
        if metrics is None:
            logger.debug("Failed to calculate performance metrics")
            return None
        
        # Calculate optimization score
        optimization_score = calculate_optimization_score(metrics, {})
        
        return {
            'parameters': params.copy(),
            'metrics': metrics,
            'optimization_score': optimization_score,
            'total_return': metrics.get('total_return', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'num_trades': metrics.get('num_trades', 0),
            'avg_trade_duration': metrics.get('avg_trade_duration', 0.0),
        }
        
    except Exception as e:
        logger.debug(f"Error testing parameter combination: {str(e)}")
        return None


def calculate_performance_metrics(results, initial_capital):
    """Calculate comprehensive performance metrics from classification results"""
    try:
        if results.empty:
            return None
        
        # Basic statistics
        total_trades = len(results)
        if total_trades == 0:
            return None
        
        # Calculate returns
        returns = results['pnl'].fillna(0)
        total_return = (returns.sum() / initial_capital) * 100
        
        # Win rate
        winning_trades = len(results[results['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = ((cumulative_returns - running_max) / running_max * 100).fillna(0)
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average trade duration (if available)
        if 'duration' in results.columns:
            avg_trade_duration = results['duration'].mean()
        else:
            avg_trade_duration = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': total_trades,
            'avg_trade_duration': avg_trade_duration,
            'total_pnl': returns.sum(),
            'avg_win': results[results['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': results[results['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0,
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return None


def calculate_optimization_score(metrics, objectives):
    """Calculate optimization score based on multiple objectives"""
    try:
        if not metrics:
            return 0.0
        
        # Base score from total return
        return_score = metrics.get('total_return', 0.0)
        
        # Penalty for high drawdown
        drawdown_penalty = metrics.get('max_drawdown', 0.0) * 0.5
        
        # Bonus for good win rate
        win_rate_bonus = metrics.get('win_rate', 0.0) * 0.2
        
        # Bonus for good Sharpe ratio
        sharpe_bonus = metrics.get('sharpe_ratio', 0.0) * 10
        
        # Penalty for too few trades
        num_trades = metrics.get('num_trades', 0)
        trade_penalty = max(0, (10 - num_trades) * 2) if num_trades < 10 else 0
        
        # Combined score
        score = return_score - drawdown_penalty + win_rate_bonus + sharpe_bonus - trade_penalty
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating optimization score: {str(e)}")
        return 0.0


def generate_parameter_combinations(config):
    """Generate parameter combinations for testing"""
    # Set maxBarsBack dynamically if not set
    if not config.param_ranges['maxBarsBack']:
        # Will be set when we have the data
        config.param_ranges['maxBarsBack'] = [100, 200, 500, 1000, 1500, 2000]
    
    # Generate all possible combinations
    param_names = list(config.param_ranges.keys())
    param_values = [config.param_ranges[param] for param in param_names]
    
    # Filter out empty parameter ranges
    filtered_names = []
    filtered_values = []
    for name, values in zip(param_names, param_values):
        if values:  # Only include non-empty parameter ranges
            filtered_names.append(name)
            filtered_values.append(values)
    
    # Generate combinations
    combinations = []
    for combo in itertools.product(*filtered_values):
        param_dict = dict(zip(filtered_names, combo))
        combinations.append(param_dict)
    
    # Shuffle for randomization
    if config.random_seed:
        np.random.seed(int(config.random_seed))
    
    np.random.shuffle(combinations)
    
    # Limit to max_combinations
    if len(combinations) > config.max_combinations:
        combinations = combinations[:config.max_combinations]
        logger.info(f"Limited to {config.max_combinations} combinations out of {len(combinations)} total")
    
    return combinations


def generate_latin_hypercube_sample(param_ranges, n_samples):
    """Generate Latin Hypercube sampling for parameter combinations"""
    try:
        from scipy.stats import qmc
        
        # Filter non-empty parameter ranges
        filtered_ranges = {k: v for k, v in param_ranges.items() if v}
        
        if not filtered_ranges:
            return []
        
        # Create sampler
        sampler = qmc.LatinHypercube(d=len(filtered_ranges))
        sample = sampler.random(n=n_samples)
        
        # Convert to parameter combinations
        combinations = []
        param_names = list(filtered_ranges.keys())
        
        for i in range(n_samples):
            param_dict = {}
            for j, param_name in enumerate(param_names):
                param_values = filtered_ranges[param_name]
                # Map [0,1] to parameter index
                index = int(sample[i, j] * len(param_values))
                index = min(index, len(param_values) - 1)  # Ensure within bounds
                param_dict[param_name] = param_values[index]
            combinations.append(param_dict)
        
        return combinations
        
    except ImportError:
        logger.warning("scipy not available, falling back to random sampling")
        return []


def display_optimization_summary(results, config, optimize_for_real_trading=False):
    """Display optimization results summary"""
    if not results:
        logger.info("No results to display")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZATION SUMMARY")
    logger.info(f"{'='*60}")
    
    # Sort by optimization score
    sorted_results = sorted(results, key=lambda x: x['optimization_score'], reverse=True)
    
    # Display top 10 results
    logger.info(f"Top 10 Parameter Combinations:")
    logger.info(f"{'Rank':<4} {'Score':<8} {'Return':<8} {'Win Rate':<8} {'Drawdown':<8} {'Trades':<6}")
    logger.info(f"{'-'*50}")
    
    for i, result in enumerate(sorted_results[:10]):
        logger.info(f"{i+1:<4} {result['optimization_score']:<8.2f} "
                   f"{result['total_return']:<8.1f}% {result['win_rate']:<8.1f}% "
                   f"{result['max_drawdown']:<8.1f}% {result['num_trades']:<6}")
    
    # Best parameters
    best_result = sorted_results[0]
    logger.info(f"\nBest Parameters:")
    for param, value in best_result['parameters'].items():
        logger.info(f"  {param}: {value}")


def save_optimization_results(results, config, optimize_for_real_trading=False):
    """Save optimization results to files"""
    if not results:
        logger.warning("No results to save")
        return
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_logs")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{config.symbol}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Prepare data for saving
    save_data = {
        'timestamp': timestamp,
        'config': {
            'symbol': config.symbol,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'max_combinations': config.max_combinations,
            'optimize_for_real_trading': optimize_for_real_trading
        },
        'results': results
    }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {filepath}")


def load_existing_best_parameters(symbol):
    """Load existing best parameters for comparison"""
    try:
        best_params_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            f"best_parameters_{symbol}.json"
        )
        
        if os.path.exists(best_params_file):
            with open(best_params_file, 'r') as f:
                return json.load(f)
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading existing best parameters: {str(e)}")
        return None


def save_best_parameters(symbol, parameters, metrics, optimization_info):
    """Save best parameters to file"""
    try:
        best_params_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            f"best_parameters_{symbol}.json"
        )
        
        # Convert numpy types to native Python types
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
        
        # Prepare data
        best_params_data = {
            'symbol': symbol,
            'parameters': convert_numpy_types(parameters),
            'metrics': convert_numpy_types(metrics),
            'optimization_info': convert_numpy_types(optimization_info),
            'optimization_date': datetime.now().isoformat()
        }
        
        # Save to file
        with open(best_params_file, 'w') as f:
            json.dump(best_params_data, f, indent=2)
        
        logger.info(f"Best parameters saved to: {best_params_file}")
        
    except Exception as e:
        logger.error(f"Error saving best parameters: {str(e)}")


def load_lumibot_parameters(symbol):
    """Load parameters from Lumibot strategy file for comparison"""
    try:
        # Try to load from AdvancedLorentzianStrategy
        strategy_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "AdvancedLorentzianStrategy.py"
        )
        
        if os.path.exists(strategy_file):
            # Read the strategy file and extract parameters
            with open(strategy_file, 'r') as f:
                content = f.read()
            
            # Simple parameter extraction (could be improved)
            # This is a basic implementation
            lumibot_params = {
                'neighborsCount': 8,  # Default values
                'maxBarsBack': 2000,
                'useDynamicExits': True,
                'rsi_period': 14,
                'rsi_smooth': 1,
                'wt_n1': 10,
                'wt_n2': 11,
                'cci_period': 20,
                'cci_smooth': 1,
                'useEmaFilter': True,
                'emaPeriod': 200,
                'useSmaFilter': False,
                'smaPeriod': 200,
                'useVolatilityFilter': True,
                'useRegimeFilter': True,
                'useAdxFilter': True,
                'regimeThreshold': -0.1,
                'adxThreshold': 20,
                'useKernelSmoothing': True,
                'lookbackWindow': 8,
                'relativeWeight': 8.0,
                'regressionLevel': 25,
                'crossoverLag': 2,
            }
            
            return lumibot_params
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading Lumibot parameters: {str(e)}")
        return None


def monitor_system_resources():
    """Monitor and display system resource usage"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free = disk.free / (1024**3)  # GB
        
        logger.info(f"System Resources:")
        logger.info(f"  CPU: {cpu_percent:.1f}%")
        logger.info(f"  Memory: {memory_percent:.1f}% ({memory_available:.1f} GB free)")
        logger.info(f"  Disk: {disk_percent:.1f}% ({disk_free:.1f} GB free)")
        
        # Warning if resources are low
        if cpu_percent > 80:
            logger.warning("High CPU usage detected")
        if memory_percent > 80:
            logger.warning("High memory usage detected")
        if disk_percent > 90:
            logger.warning("Low disk space detected")
        
    except Exception as e:
        logger.error(f"Error monitoring system resources: {str(e)}")


def validate_optimization_vs_strategy(config):
    """Validate optimization configuration against strategy requirements"""
    try:
        # Check for required parameters
        required_params = ['neighborsCount', 'maxBarsBack', 'useDynamicExits']
        for param in required_params:
            if param not in config.param_ranges or not config.param_ranges[param]:
                logger.error(f"Missing required parameter: {param}")
                return False
        
        # Check parameter ranges are reasonable
        if max(config.param_ranges['neighborsCount']) > 100:
            logger.warning("Very high neighborsCount values may cause performance issues")
        
        # Check data requirements
        if config.max_data_set < 100:
            logger.error("MAX_DATA_SET too small, need at least 100 bars")
            return False
        
        logger.info("Optimization configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating optimization configuration: {str(e)}")
        return False


def validate_best_parameters_on_backtest_data(results, config):
    """Validate best parameters on separate backtest data"""
    try:
        if not results:
            logger.warning("No results to validate")
            return
        
        # Get best parameters
        best_result = max(results, key=lambda x: x['optimization_score'])
        best_params = best_result['parameters']
        
        # Test on backtest data if available
        if hasattr(config, 'backtest_data') and config.backtest_data is not None:
            logger.info("Testing best parameters on backtest data...")
            
            validation_result = test_parameter_combination(
                best_params, 
                config.backtest_data, 
                config.symbol, 
                config.initial_capital
            )
            
            if validation_result:
                logger.info(f"Backtest validation results:")
                logger.info(f"  Return: {validation_result['total_return']:+.1f}%")
                logger.info(f"  Win Rate: {validation_result['win_rate']:.1f}%")
                logger.info(f"  Max Drawdown: {validation_result['max_drawdown']:.1f}%")
                logger.info(f"  Trades: {validation_result['num_trades']}")
            else:
                logger.warning("Backtest validation failed")
        
    except Exception as e:
        logger.error(f"Error validating best parameters: {str(e)}")


def perform_walk_forward_analysis(param_combinations, df, config):
    """Perform walk-forward analysis on parameter combinations"""
    try:
        if not param_combinations:
            logger.warning("No parameter combinations for walk-forward analysis")
            return
        
        logger.info(f"Starting walk-forward analysis with {len(param_combinations)} parameter sets")
        
        # Split data into periods
        periods = config.walk_forward_periods
        period_size = len(df) // periods
        
        results = []
        
        for i, params in enumerate(param_combinations[:5]):  # Test top 5
            logger.info(f"Testing parameter set {i+1}/{min(5, len(param_combinations))}")
            
            period_results = []
            
            for period in range(periods):
                start_idx = period * period_size
                end_idx = min(start_idx + period_size, len(df))
                
                period_df = df.iloc[start_idx:end_idx].copy()
                
                # Test on this period
                result = test_parameter_combination(
                    params, 
                    period_df, 
                    config.symbol, 
                    config.initial_capital
                )
                
                if result:
                    period_results.append(result)
            
            # Calculate average performance across periods
            if period_results:
                avg_return = np.mean([r['total_return'] for r in period_results])
                avg_win_rate = np.mean([r['win_rate'] for r in period_results])
                
                results.append({
                    'parameters': params,
                    'avg_return': avg_return,
                    'avg_win_rate': avg_win_rate,
                    'period_results': period_results
                })
        
        # Display walk-forward results
        if results:
            logger.info("\nWalk-Forward Analysis Results:")
            logger.info(f"{'Rank':<4} {'Avg Return':<12} {'Avg Win Rate':<12} {'Stability':<10}")
            logger.info(f"{'-'*50}")
            
            for i, result in enumerate(sorted(results, key=lambda x: x['avg_return'], reverse=True)):
                stability = np.std([r['total_return'] for r in result['period_results']])
                logger.info(f"{i+1:<4} {result['avg_return']:<12.1f}% {result['avg_win_rate']:<12.1f}% {stability:<10.1f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in walk-forward analysis: {str(e)}")
        return []


def validate_environment_configuration():
    """Validate environment configuration before starting optimization"""
    try:
        logger.info("Validating environment configuration...")
        
        # Check required environment variables
        required_vars = ['POLYGON_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        # Check optional but recommended variables
        optional_vars = ['SYMBOL', 'INITIAL_CAPITAL', 'MAX_COMBINATIONS']
        for var in optional_vars:
            if not os.getenv(var):
                logger.info(f"Using default value for {var}")
        
        # Check system resources
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 2:
            logger.warning(f"Low available memory: {available_memory:.1f} GB")
        
        # Check disk space
        disk_free = psutil.disk_usage('/').free / (1024**3)  # GB
        if disk_free < 1:
            logger.warning(f"Low disk space: {disk_free:.1f} GB")
        
        logger.info("Environment configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating environment configuration: {str(e)}")
        return False


def main():
    """Main optimization function"""
    logger.info("Starting Parameter Optimization for Lorentzian Classification")
    logger.info("="*80)
    
    # Validate configuration before proceeding
    if not validate_environment_configuration():
        logger.error("Exiting due to configuration errors")
        sys.exit(1)
    
    # Check system resources before starting
    monitor_system_resources()
    
    # Load configuration
    config = OptimizationConfig()
    
    logger.info(f"Configuration:")
    logger.info(f"  Symbol: {config.symbol}")
    logger.info(f"  Period: {config.start_date} to {config.end_date}")
    logger.info(f"  Timeframe: {config.timeframe}")
    logger.info(f"  Capital: ${config.initial_capital:,.0f}")
    logger.info(f"  Max combinations: {config.max_combinations:,}")
    logger.info(f"  Parallel: {'enabled' if config.use_parallel else 'disabled'}")
    logger.info(f"  Cores: {config.n_jobs}")
    logger.info(f"  Seed: {config.random_seed if config.random_seed else 'random'}")
    
    # Display optimization strategy
    strategy_name = "Return-Focused" if config.optimize_for_return else "Balanced"
    logger.info(f"  Strategy: {strategy_name}")
    
    # Validate timeframe
    if config.timeframe not in ['day', 'hour', 'minute']:
        logger.warning(f"Invalid timeframe '{config.timeframe}', using 'day'")
        config.timeframe = 'day'
    
    # Warning for intraday data optimization
    if config.timeframe in ['minute', 'hour']:
        start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        if config.timeframe == 'minute' and days_diff > 7:
            logger.warning(f"{days_diff} days of minute data will be VERY slow")
        elif config.timeframe == 'hour' and days_diff > 90:
            logger.warning(f"{days_diff} days of hourly data may be slow")
    
    # Set default maxBarsBack range for validation (will be updated after data loading)
    if not config.param_ranges['maxBarsBack']:
        config.param_ranges['maxBarsBack'] = [100, 500, 1000, 2000]
    
    # Validate optimization vs strategy (after setting up basic param ranges)
    if not validate_optimization_vs_strategy(config):
        return
    
    # Interactive optimization method selection
    logger.info(f"\nOptimization Method Selection")
    logger.info(f"=" * 40)
    logger.info(f"Choose your optimization approach:")
    logger.info(f"")
    logger.info(f"1. Backtesting Research")
    logger.info(f"   - Test on specific historical period")
    logger.info(f"   - Train on data BEFORE backtest period")
    logger.info(f"   - Eliminates look-ahead bias")
    logger.info(f"")
    logger.info(f"2. Live Trading")
    logger.info(f"   - Find best parameters for deployment")
    logger.info(f"   - Use full 2-year data window")
    logger.info(f"   - Maximum training data")
    logger.info(f"")
    
    while True:
        choice = input(f"Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        logger.error("Invalid choice. Please enter 1 or 2.")
    
    optimize_for_real_trading = (choice == '2')
    method_name = "Live Trading" if optimize_for_real_trading else "Backtesting Research"
    logger.info(f"Selected: {method_name}")
    
    # Download market data based on selected method
    if optimize_for_real_trading:
        # METHOD 2: Use full data window for live trading optimization
        logger.info("\nDownloading full data window for live trading...")
        
        # Get all available data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        logger.info(f"  Data window: {start_date} to {end_date}")
        
        # Download all available data
        df_full = download_real_data(
            symbol=config.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=config.timeframe,
        )
        
        if df_full.empty:
            logger.error("Failed to download market data")
            return
        
        # Use all data for optimization
        df_for_classification = df_full
        
        logger.info(f"  Downloaded {len(df_for_classification)} bars for optimization")
        
    else:
        # METHOD 1: Separate training and backtesting data
        logger.info("\nDownloading separated training and backtesting data...")
        
        # Calculate training period (before backtest period)
        backtest_start = datetime.strptime(config.start_date, '%Y-%m-%d')
        training_end = backtest_start - timedelta(days=1)
        training_start = training_end - timedelta(days=config.max_data_set)
        
        logger.info(f"  Training data: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}")
        logger.info(f"  Backtest data: {config.start_date} to {config.end_date}")
        
        # Download training data
        df_training = download_real_data(
            symbol=config.symbol,
            start_date=training_start.strftime('%Y-%m-%d'),
            end_date=training_end.strftime('%Y-%m-%d'),
            timeframe=config.timeframe,
        )
        
        # Download backtest data
        df_backtest = download_real_data(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
        )
        
        if df_training.empty or df_backtest.empty:
            logger.error("Failed to download market data")
            return
        
        # Use training data for optimization
        df_for_classification = df_training
        config.backtest_data = df_backtest
        
        logger.info(f"  Downloaded {len(df_for_classification)} training bars")
        logger.info(f"  Downloaded {len(df_backtest)} backtest bars")
    
    # Aggregate data if needed
    if config.aggregate_to_daily and config.timeframe != 'day':
        logger.info(f"Aggregating {config.timeframe} data to daily...")
        
        if config.timeframe == 'minute':
            df_for_classification = aggregate_minute_to_daily(df_for_classification)
        elif config.timeframe == 'hour':
            df_for_classification = aggregate_hour_to_daily(df_for_classification)
        
        logger.info(f"  Aggregated to {len(df_for_classification)} daily bars")
    
    # Set maxBarsBack dynamically based on actual data
    max_bars_back = min(config.max_data_set, len(df_for_classification) - 50)
    config.param_ranges['maxBarsBack'] = [
        max_bars_back // 4,
        max_bars_back // 2,
        int(max_bars_back * 0.75),
        max_bars_back
    ]
    logger.info(f"Updated maxBarsBack options based on data: {config.param_ranges['maxBarsBack']}")
    
    # Initialize smart optimizer
    smart_optimizer = None
    if config.use_smart_optimization:
        smart_optimizer = SmartOptimizer(config)
        smart_optimizer.load_state()
    
    # Generate parameter combinations
    if smart_optimizer:
        logger.info("Generating smart parameter combinations...")
        combinations = smart_optimizer.generate_smart_combinations(config.max_combinations)
    else:
        logger.info("Generating parameter combinations...")
        combinations = generate_parameter_combinations(config)
    
    # Filter out already tested combinations
    if smart_optimizer:
        original_count = len(combinations)
        combinations = [c for c in combinations if not smart_optimizer.is_combination_tested(c)]
        logger.info(f"Filtered out {original_count - len(combinations)} already tested combinations")
    
    logger.info(f"Testing {len(combinations)} parameter combinations...")
    
    # Test parameter combinations
    results = []
    
    if config.use_parallel and config.n_jobs > 1:
        logger.info(f"Using parallel processing with {config.n_jobs} cores")
        
        # Create partial function for multiprocessing
        test_func = partial(
            test_parameter_combination_wrapper,
            df=df_for_classification,
            symbol=config.symbol,
            initial_capital=config.initial_capital,
            objectives={},
            max_bars_back=max_bars_back
        )
        
        # Use multiprocessing
        with mp.Pool(processes=config.n_jobs) as pool:
            with tqdm(total=len(combinations), desc="Testing combinations") as pbar:
                for result in pool.imap(test_func, combinations):
                    if result:
                        results.append(result)
                        
                        # Update smart optimizer
                        if smart_optimizer:
                            smart_optimizer.mark_combination_tested(result['parameters'], result)
                    
                    pbar.update(1)
    else:
        logger.info("Using sequential processing")
        
        # Sequential processing
        for i, params in enumerate(tqdm(combinations, desc="Testing combinations")):
            result = test_parameter_combination(
                params,
                df_for_classification,
                config.symbol,
                config.initial_capital,
                max_bars_back
            )
            
            if result:
                results.append(result)
                
                # Update smart optimizer
                if smart_optimizer:
                    smart_optimizer.mark_combination_tested(params, result)
            
            # Progress update
            if (i + 1) % 50 == 0:
                logger.info(f"Completed {i + 1}/{len(combinations)} combinations")
    
    # Process results
    if results:
        logger.info(f"\nOptimization completed with {len(results)} valid results")
        
        # Display summary
        display_optimization_summary(results, config, optimize_for_real_trading)
        
        # Save results
        save_optimization_results(results, config, optimize_for_real_trading)
        
        # Find and save best parameters
        best_result = max(results, key=lambda x: x['optimization_score'])
        
        # Load existing best parameters for comparison
        existing_best = load_existing_best_parameters(config.symbol)
        
        # Compare with existing best
        current_score = best_result['optimization_score']
        should_save = True
        
        if existing_best:
            existing_score = existing_best.get('optimization_info', {}).get('optimization_score', 0)
            
            if current_score > existing_score:
                logger.info(f"NEW BEST FOUND! Current: {current_score:.3f} vs Previous: {existing_score:.3f}")
            else:
                logger.info(f"Previous best is still optimal: {existing_score:.3f} vs {current_score:.3f}")
                should_save = False
        
        # Save best parameters
        if should_save:
            save_best_parameters(
                config.symbol,
                best_result['parameters'],
                best_result['metrics'],
                {
                    'optimization_score': best_result['optimization_score'],
                    'method': method_name,
                    'data_period': f"{config.start_date} to {config.end_date}" if not optimize_for_real_trading else f"{training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}"
                }
            )
        
        # Display detailed report
        logger.info("\nDetailed Report - Best Parameters:")
        display_performance_report(best_result)
        
        # Validate best parameters on actual backtest data (only for Method 1)
        if hasattr(config, 'backtest_data') and config.backtest_data is not None:
            logger.info("\nValidating parameters on backtest data...")
            validate_best_parameters_on_backtest_data(results, config)
        elif optimize_for_real_trading:
            logger.info("\nLive trading optimization complete")
            logger.info("Parameters ready for AdvancedLorentzianStrategy")
        
        # Perform walk-forward analysis if enabled
        if config.use_walk_forward and len(results) >= 5:
            logger.info("\nStarting walk-forward analysis...")
            results.sort(key=lambda x: x['optimization_score'], reverse=True)
            top_param_combinations = [r['parameters'] for r in results[:10]]
            walk_forward_results = perform_walk_forward_analysis(top_param_combinations, df_for_classification, config)
        elif config.use_walk_forward:
            logger.info(f"Skipping walk-forward: need 5+ results, got {len(results)}")
        
    else:
        logger.error("No valid results found. Check parameter ranges and data.")
    
    # Save optimization state for next run
    if smart_optimizer:
        smart_optimizer.save_state()
        logger.info("Saved optimization state for future runs")


if __name__ == "__main__":
    main()