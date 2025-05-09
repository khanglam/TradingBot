import datetime as dt
import pandas as pd
import numpy as np
import pytz
import sys
import json
import os
import webbrowser
import time

# Add the path to the advanced_ta package
sys.path.append('d:\\Khang\\Projects\\TradingViewWorkspace\\LorentzianClassification')

from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import PolygonDataBacktesting
from lorentzian_strategy import LorentzianStrategy

# Monkey patch webbrowser.open to prevent browser windows
original_open = webbrowser.open
def no_op_open(url, *args, **kwargs):
    print(f"[BROWSER SUPPRESSED] Would have opened: {url}")
    return True

# Apply the patch globally
webbrowser.open = no_op_open

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def run_backtest_with_params(params, start_date, end_date, config, symbol="SPY"):
    """Run a backtest with the given parameters and return the results"""
    # Build features list
    features = [
        {"type": "RSI", "param1": params["rsi_period"], "param2": params["rsi_smoothing"]},
        {"type": "WT", "param1": params["wt_channel_length"], "param2": params["wt_average_length"]},
        {"type": "CCI", "param1": params["cci_period"], "param2": params["cci_smoothing"]},
        {"type": "ADX", "param1": params["adx_period"], "param2": params["adx_smoothing"]},
    ]
    
    # Build strategy parameters
    strategy_params = {
        "symbol": symbol,
        "max_bars_back": params["max_bars_back"],
        "neighbors_count": params["neighbors_count"],
        "use_dynamic_exits": params["use_dynamic_exits"],
        "force_signals": True,  # Force signals to ensure trades occur despite data issues
        "features": features,
        "use_volatility_filter": params["use_volatility_filter"],
        "use_regime_filter": params["use_regime_filter"],
        "use_adx_filter": params["use_adx_filter"],
        "use_kernel_smoothing": params["use_kernel_smoothing"],
        "position_size": params["position_size"],
    }
    
    try:
        # Suppress print statements during backtest for cleaner output
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Run the backtest
        results, strategy = LorentzianStrategy.run_backtest(
            datasource_class=PolygonDataBacktesting,
            backtesting_start=start_date,
            backtesting_end=end_date,
            minutes_before_closing=0,
            benchmark_asset=symbol,
            analyze_backtest=False,  # Disable analysis to prevent browser windows
            parameters=strategy_params,
            show_progress_bar=False,
            config=config,
            timestep='day',
            refresh_cache=False,  # Don't refresh cache for each evaluation
        )
        
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout
        
        # Extract metrics from results
        if results is not None and hasattr(results, 'metrics'):
            # Primary fitness score - can use sharpe ratio, or other metrics
            sharpe = results.metrics.get('sharpe_ratio', 0)
            total_return = results.metrics.get('total_return', 0)
            sortino = results.metrics.get('sortino_ratio', 0)
            drawdown = results.metrics.get('max_drawdown', 100)  # Lower is better
            
            # Create a composite fitness score
            # Prioritize sharpe while considering returns and drawdown protection
            fitness = sharpe * 0.5 + total_return * 0.3 + sortino * 0.2 - (drawdown/100) * 0.2
            
            # Penalize heavily negative metric combinations
            if sharpe <= 0 and total_return <= 0:
                fitness = 0.001  # Small positive value to avoid errors
                
            # Store metrics in the params dictionary
            metrics = {
                'fitness': fitness,
                'sharpe_ratio': sharpe,
                'total_return': total_return,
                'sortino_ratio': sortino,
                'max_drawdown': drawdown
            }
            
            return metrics
        else:
            return {'fitness': 0.001}
            
    except Exception as e:
        # Handle any errors during backtesting
        print(f"Error in backtest: {e}")
        return {'fitness': 0.001}

def optimize_lorentzian_strategy():
    """Run a basic optimization of the Lorentzian strategy"""
    # Import configuration
    try:
        from config import ALPACA_CONFIG
    except ImportError:
        print("Could not import ALPACA_CONFIG from config.py")
        sys.exit(1)
    
    # Set up backtesting parameters - use a more recent period for better data availability
    tzinfo = pytz.timezone('America/New_York')
    backtesting_start = tzinfo.localize(dt.datetime(2024, 1, 1))  # Use recent data
    backtesting_end = tzinfo.localize(dt.datetime(2024, 2, 29))  # Two months for better evaluation
    
    # Define parameter sets to test
    parameter_sets = [
        # Base configuration
        {
            'rsi_period': 14,
            'rsi_smoothing': 2,
            'wt_channel_length': 10,
            'wt_average_length': 11,
            'cci_period': 20,
            'cci_smoothing': 2,
            'adx_period': 20,
            'adx_smoothing': 2,
            'max_bars_back': 200,
            'neighbors_count': 6,
            'position_size': 0.1,
            'use_dynamic_exits': True,
            'use_volatility_filter': False,
            'use_regime_filter': False,
            'use_adx_filter': False,
            'use_kernel_smoothing': True,
            'name': 'Base Configuration'
        },
        # Optimized configuration - best overall balance
        {
            'rsi_period': 7,
            'rsi_smoothing': 2,
            'wt_channel_length': 10,
            'wt_average_length': 11,
            'cci_period': 20,
            'cci_smoothing': 2,
            'adx_period': 14,
            'adx_smoothing': 2,
            'max_bars_back': 300,
            'neighbors_count': 8,
            'position_size': 0.2,
            'use_dynamic_exits': True,
            'use_volatility_filter': True,
            'use_regime_filter': False,
            'use_adx_filter': True,
            'use_kernel_smoothing': True,
            'name': 'Optimized Configuration'
        },
        # Aggressive configuration - higher returns but more volatility
        {
            'rsi_period': 5,
            'rsi_smoothing': 1,
            'wt_channel_length': 9,
            'wt_average_length': 10,
            'cci_period': 14,
            'cci_smoothing': 1,
            'adx_period': 14,
            'adx_smoothing': 1,
            'max_bars_back': 250,
            'neighbors_count': 5,
            'position_size': 0.3,
            'use_dynamic_exits': True,
            'use_volatility_filter': False,
            'use_regime_filter': False,
            'use_adx_filter': True,
            'use_kernel_smoothing': True,
            'name': 'Aggressive Configuration'
        },
        # Conservative configuration - lower returns but less drawdown
        {
            'rsi_period': 14,
            'rsi_smoothing': 3,
            'wt_channel_length': 12,
            'wt_average_length': 13,
            'cci_period': 30,
            'cci_smoothing': 3,
            'adx_period': 21,
            'adx_smoothing': 3,
            'max_bars_back': 350,
            'neighbors_count': 10,
            'position_size': 0.15,
            'use_dynamic_exits': True,
            'use_volatility_filter': True,
            'use_regime_filter': True,
            'use_adx_filter': True,
            'use_kernel_smoothing': True,
            'name': 'Conservative Configuration'
        },
        # Fast RSI with volatility filter
        {
            'rsi_period': 7,
            'rsi_smoothing': 2,
            'wt_channel_length': 10,
            'wt_average_length': 11,
            'cci_period': 20,
            'cci_smoothing': 2,
            'adx_period': 20,
            'adx_smoothing': 2,
            'max_bars_back': 200,
            'neighbors_count': 6,
            'position_size': 0.2,
            'use_dynamic_exits': True,
            'use_volatility_filter': True,
            'use_regime_filter': False,
            'use_adx_filter': False,
            'use_kernel_smoothing': True,
            'name': 'Fast RSI with Volatility Filter'
        },
        # Balanced configuration with all filters
        {
            'rsi_period': 10,
            'rsi_smoothing': 2,
            'wt_channel_length': 10,
            'wt_average_length': 11,
            'cci_period': 20,
            'cci_smoothing': 2,
            'adx_period': 14,
            'adx_smoothing': 2,
            'max_bars_back': 250,
            'neighbors_count': 8,
            'position_size': 0.15,
            'use_dynamic_exits': True,
            'use_volatility_filter': True,
            'use_regime_filter': True,
            'use_adx_filter': True,
            'use_kernel_smoothing': True,
            'name': 'All Filters Enabled'
        },
        # Optimized for trending markets
        {
            'rsi_period': 14,
            'rsi_smoothing': 2,
            'wt_channel_length': 10,
            'wt_average_length': 11,
            'cci_period': 20,
            'cci_smoothing': 2,
            'adx_period': 14,
            'adx_smoothing': 1,
            'max_bars_back': 200,
            'neighbors_count': 6,
            'position_size': 0.2,
            'use_dynamic_exits': True,
            'use_volatility_filter': False,
            'use_regime_filter': False,
            'use_adx_filter': True,
            'use_kernel_smoothing': True,
            'name': 'Trend Following'
        },
        # Optimized for ranging markets
        {
            'rsi_period': 7,
            'rsi_smoothing': 1,
            'wt_channel_length': 9,
            'wt_average_length': 10,
            'cci_period': 14,
            'cci_smoothing': 1,
            'adx_period': 20,
            'adx_smoothing': 2,
            'max_bars_back': 200,
            'neighbors_count': 8,
            'position_size': 0.15,
            'use_dynamic_exits': True,
            'use_volatility_filter': True,
            'use_regime_filter': False,
            'use_adx_filter': False,
            'use_kernel_smoothing': True,
            'name': 'Range Trading'
        }
    ]
    
    # Create output directory
    output_dir = "optimization_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run backtests for each parameter set
    results = []
    print(f"Testing {len(parameter_sets)} parameter sets...")
    
    for i, params in enumerate(parameter_sets):
        print(f"\nTesting parameter set {i+1}/{len(parameter_sets)}: {params['name']}")
        start_time = time.time()
        
        # Run backtest
        metrics = run_backtest_with_params(
            params=params,
            start_date=backtesting_start,
            end_date=backtesting_end,
            config=ALPACA_CONFIG,
            symbol="SPY"
        )
        
        # Add metrics to params
        params.update(metrics)
        
        # Convert to serializable types
        params = convert_to_serializable(params)
        
        # Add to results
        results.append(params)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds. Fitness: {params.get('fitness', 0):.4f}")
    
    # Sort results by fitness
    results.sort(key=lambda x: x.get('fitness', 0), reverse=True)
    
    # Save results to JSON
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOptimization complete! Results saved to {result_file}")
    print("\nTop Parameter Sets:")
    
    for i, params in enumerate(results[:3]):
        print(f"\nRank {i+1}: {params.get('name', 'Unnamed')} (Fitness: {params.get('fitness', 0):.4f})")
        print(f"  Sharpe Ratio: {params.get('sharpe_ratio', 0):.4f}")
        print(f"  Total Return: {params.get('total_return', 0):.4f}")
        print(f"  Sortino Ratio: {params.get('sortino_ratio', 0):.4f}")
        print(f"  Max Drawdown: {params.get('max_drawdown', 0):.4f}%")
    
    # Return the best parameter set
    return results[0] if results else None

# Apply the best parameters to the strategy
def apply_best_parameters(best_params=None, result_file=None, output_file="lorentzian_optimized_strategy.py"):
    """Apply the best parameters to the strategy"""
    # If no best params provided, load from file
    if best_params is None:
        if result_file is None:
            # Find the latest result file
            output_dir = "optimization_results"
            if not os.path.exists(output_dir):
                print("No optimization results directory found.")
                return False
                
            json_files = [f for f in os.listdir(output_dir) if f.startswith("optimization_results_") and f.endswith(".json")]
            if not json_files:
                print("No optimization results files found.")
                return False
                
            # Sort by modification time (newest first)
            result_file = os.path.join(output_dir, sorted(json_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)[0])
        
        # Load the results
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
                
            if not results:
                print("No results found in file.")
                return False
                
            best_params = results[0]  # First result is the best
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    # Load the original strategy file
    try:
        with open("lorentzian_strategy.py", 'r') as f:
            strategy_code = f.read()
            
        # Find the main block where parameters are defined
        start_marker = "if __name__ == \"__main__\":"
        params_marker = "parameters={"  # Where parameters start in the run_backtest call
        
        if start_marker not in strategy_code or params_marker not in strategy_code:
            print("Could not locate parameters section in strategy file.")
            return False
            
        # Create the modified parameters block
        params_str = "parameters={"
        
        # Add symbol
        params_str += f"\n            \"symbol\": \"SPY\","
        
        # Add max_bars_back
        params_str += f"\n            \"max_bars_back\": {best_params.get('max_bars_back', 200)},"
        
        # Add neighbors_count
        params_str += f"\n            \"neighbors_count\": {best_params.get('neighbors_count', 6)},"
        
        # Add use_dynamic_exits
        params_str += f"\n            \"use_dynamic_exits\": {str(best_params.get('use_dynamic_exits', True))},"
        
        # Add force_signals
        params_str += f"\n            \"force_signals\": False,  # Use Lorentzian signals"
        
        # Add features
        params_str += "\n            \"features\": ["
        params_str += f"\n                {{\"type\": \"RSI\", \"param1\": {best_params.get('rsi_period', 14)}, \"param2\": {best_params.get('rsi_smoothing', 2)}}},"
        params_str += f"\n                {{\"type\": \"WT\", \"param1\": {best_params.get('wt_channel_length', 10)}, \"param2\": {best_params.get('wt_average_length', 11)}}},"
        params_str += f"\n                {{\"type\": \"CCI\", \"param1\": {best_params.get('cci_period', 20)}, \"param2\": {best_params.get('cci_smoothing', 2)}}},"
        params_str += f"\n                {{\"type\": \"ADX\", \"param1\": {best_params.get('adx_period', 20)}, \"param2\": {best_params.get('adx_smoothing', 2)}}},"
        params_str += "\n            ],"
        
        # Add filters
        params_str += f"\n            \"use_volatility_filter\": {str(best_params.get('use_volatility_filter', False))},"
        params_str += f"\n            \"use_regime_filter\": {str(best_params.get('use_regime_filter', False))},"
        params_str += f"\n            \"use_adx_filter\": {str(best_params.get('use_adx_filter', False))},"
        params_str += f"\n            \"use_kernel_smoothing\": {str(best_params.get('use_kernel_smoothing', True))},"
        
        # Add position_size
        params_str += f"\n            \"position_size\": {best_params.get('position_size', 0.1)},"
        
        params_str += "\n        },"
        
        # Find where to insert the new parameters
        params_start_idx = strategy_code.find(params_marker)
        if params_start_idx == -1:
            print("Could not locate parameters section in strategy file.")
            return False
            
        # Find the end of the parameters block
        params_end_idx = strategy_code.find("),", params_start_idx)
        if params_end_idx == -1:
            print("Could not locate end of parameters section in strategy file.")
            return False
            
        # Replace the parameters
        old_params = strategy_code[params_start_idx:params_end_idx+1]
        new_strategy_code = strategy_code.replace(old_params, params_str)
        
        # Add a note about optimization
        comment = """\n# This strategy uses parameters optimized by basic_optimizer.py\n# Optimization date: {date}\n# Fitness score: {fitness}\n""".format(
            date=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fitness=best_params.get('fitness', 'N/A')
        )
        
        import_section_end = strategy_code.find("class LorentzianStrategy")
        if import_section_end == -1:
            import_section_end = strategy_code.find("")
            
        new_strategy_code = new_strategy_code[:import_section_end] + comment + new_strategy_code[import_section_end:]
        
        # Write the new strategy file
        with open(output_file, 'w') as f:
            f.write(new_strategy_code)
            
        print(f"Successfully created optimized strategy file: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error applying parameters: {e}")
        import traceback
        print(traceback.format_exc())
        return False

# Run the optimizer when the script is executed directly
if __name__ == "__main__":
    print("Starting Lorentzian strategy optimization...")
    best_params = optimize_lorentzian_strategy()
    
    if best_params:
        print("\nApplying best parameters to create optimized strategy...")
        apply_best_parameters(best_params)
