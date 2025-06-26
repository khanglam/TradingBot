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

# Load environment variables
load_dotenv()

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our existing functions
from run_advanced_ta import (
    download_real_data, 
    calculate_performance_metrics, 
    display_performance_report
)

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

class OptimizationConfig:
    """Configuration for parameter optimization"""
    
    def __init__(self):
        # Market data settings
        self.symbol = os.getenv('SYMBOL', 'TSLA')
        self.start_date = os.getenv('BACKTESTING_START', '2024-01-31')
        self.end_date = os.getenv('BACKTESTING_END', '2024-12-31')
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
        
        # Optimization settings
        self.max_combinations = int(os.getenv('MAX_COMBINATIONS', '50'))  # Limit for testing
        
        # Parameter ranges to optimize
        self.param_ranges = {
            'neighborsCount': [4, 6, 8, 10],
            'maxBarsBack': [500, 1000, 1500],
            'useDynamicExits': [True, False],
            
            # Feature parameters
            'rsi_period': [10, 14, 18],
            'rsi_smooth': [1, 2],
            'wt_n1': [6, 9, 12],
            'wt_n2': [8, 10],
            'cci_period': [10, 14, 18],
            'cci_smooth': [1, 2],
            
            # Filter settings
            'useVolatilityFilter': [True, False],
            'useRegimeFilter': [True, False],
            'useAdxFilter': [True, False],
            
            # Kernel filter settings
            'useKernelSmoothing': [True, False],
            'lookbackWindow': [6, 8, 10],
            'relativeWeight': [6.0, 8.0, 10.0],
            'regressionLevel': [20, 25, 30],
        }
        
        # Optimization objectives
        self.objectives = {
            'total_return': {'weight': 0.4, 'direction': 'maximize'},
            'win_rate': {'weight': 0.2, 'direction': 'maximize'},
            'profit_factor': {'weight': 0.2, 'direction': 'maximize'},
            'sharpe_ratio': {'weight': 0.1, 'direction': 'maximize'},
            'max_drawdown': {'weight': 0.1, 'direction': 'minimize'},  # Lower is better
        }

def test_parameter_combination(params, df, symbol, initial_capital):
    """Test a single parameter combination and return performance metrics"""
    try:
        # Create features with optimized parameters
        features = [
            Feature("RSI", params['rsi_period'], params['rsi_smooth']),
            Feature("WT", params['wt_n1'], params['wt_n2']),
            Feature("CCI", params['cci_period'], params['cci_smooth'])
        ]
        
        # Create settings
        settings = Settings(
            source=df['close'],
            neighborsCount=params['neighborsCount'],
            maxBarsBack=params['maxBarsBack'],
            useDynamicExits=params['useDynamicExits'],
            useEmaFilter=False,  # Keep disabled for now
            emaPeriod=200,
            useSmaFilter=False,  # Keep disabled for now
            smaPeriod=200
        )
        
        # Create kernel filter
        kernel_filter = KernelFilter(
            useKernelSmoothing=params['useKernelSmoothing'],
            lookbackWindow=params['lookbackWindow'],
            relativeWeight=params['relativeWeight'],
            regressionLevel=params['regressionLevel'],
            crossoverLag=2
        )
        
        # Create filter settings
        filter_settings = FilterSettings(
            useVolatilityFilter=params['useVolatilityFilter'],
            useRegimeFilter=params['useRegimeFilter'],
            useAdxFilter=params['useAdxFilter'],
            regimeThreshold=0.0,
            adxThreshold=20,
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
        print(f"❌ Error testing parameters: {e}")
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
    """Generate parameter combinations for optimization"""
    param_names = list(config.param_ranges.keys())
    param_values = list(config.param_ranges.values())
    
    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))
    
    # Limit combinations if too many
    if len(all_combinations) > config.max_combinations:
        print(f"⚠️  Too many combinations ({len(all_combinations)}), randomly sampling {config.max_combinations}")
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(len(all_combinations), config.max_combinations, replace=False)
        all_combinations = [all_combinations[i] for i in selected_indices]
    
    # Convert to parameter dictionaries
    param_combinations = []
    for combination in all_combinations:
        params = dict(zip(param_names, combination))
        param_combinations.append(params)
    
    return param_combinations

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
        print(f"🎛️  Filters: Vol={params['useVolatilityFilter']} | Regime={params['useRegimeFilter']} | ADX={params['useAdxFilter']}")
    
    # Best parameters
    best_result = results[0]
    print(f"\n🎯 BEST PARAMETERS (Score: {best_result['optimization_score']:.3f}):")
    print("="*80)
    
    best_params = best_result['parameters']
    for param, value in best_params.items():
        print(f"   {param}: {value}")

def save_optimization_results(results, config):
    """Save optimization results to JSON file"""
    results_dir = "results_logs"
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
    
    # Create best parameters file
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
        'best_parameters': {
            # Core settings
            'neighborsCount': best_params['neighborsCount'],
            'maxBarsBack': best_params['maxBarsBack'],
            'useDynamicExits': best_params['useDynamicExits'],
            
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
                'regimeThreshold': 0.0,
                'adxThreshold': 20
            },
            
            # Kernel filter settings
            'kernel_filter': {
                'useKernelSmoothing': best_params['useKernelSmoothing'],
                'lookbackWindow': best_params['lookbackWindow'],
                'relativeWeight': best_params['relativeWeight'],
                'regressionLevel': best_params['regressionLevel'],
                'crossoverLag': 2
            }
        }
    }
    
    # Save to standardized filename
    best_params_file = f"best_parameters_{config.symbol}.json"
    with open(best_params_file, 'w') as f:
        json.dump(best_params_data, f, indent=2)
    
    print(f"🎯 Best parameters saved to: {best_params_file}")
    return best_params_file

def main():
    """Main optimization function"""
    print("🔧 Starting Parameter Optimization for Lorentzian Classification")
    print("="*80)
    
    # Load configuration
    config = OptimizationConfig()
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Date range: {config.start_date} to {config.end_date}")
    print(f"   Initial capital: ${config.initial_capital:,.2f}")
    print(f"   Max combinations: {config.max_combinations}")
    
    # Download market data
    print(f"\n📥 Downloading market data...")
    df = download_real_data(symbol=config.symbol, start_date=config.start_date, end_date=config.end_date)
    
    # Convert to lowercase columns for classification
    df_for_classification = df.copy()
    df_for_classification.columns = df_for_classification.columns.str.lower()
    
    # Generate parameter combinations
    print(f"\n🔄 Generating parameter combinations...")
    param_combinations = generate_parameter_combinations(config)
    print(f"   Testing {len(param_combinations)} parameter combinations")
    
    # Run optimization
    print(f"\n🧠 Running optimization...")
    results = []
    
    # Use progress bar
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