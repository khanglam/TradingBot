from datetime import datetime
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy
import pandas as pd
import itertools
import random


class UltraFastLorentzianOptimizer:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # ULTRA-REDUCED parameter ranges - focus on most impactful parameters
        # Based on typical ML best practices and trading experience
        self.param_grid = {
            'neighbors': [5, 8, 12],           # 3 values - k-NN sweet spot
            'history_window': [200, 300],      # 2 values - enough history vs speed
            'rsi_length': [14],                # 1 value - standard RSI period
            'wt_channel': [10],                # 1 value - standard WT period  
            'wt_average': [11],                # 1 value - standard WT average
            'cci_length': [20]                 # 1 value - standard CCI period
        }
        
        # Alternative: Random sampling approach
        self.use_random_sampling = True
        self.max_random_samples = 50  # Test only 50 random combinations
        
    def get_optimization_approach(self):
        """Choose between grid search and random sampling"""
        grid_total = 1
        for values in self.param_grid.values():
            grid_total *= len(values)
            
        if self.use_random_sampling and grid_total > self.max_random_samples:
            return "random", self.max_random_samples
        else:
            return "grid", grid_total
    
    def generate_random_params(self):
        """Generate random parameter combinations"""
        # Expanded ranges for random sampling
        random_ranges = {
            'neighbors': list(range(3, 21)),           # 3-20
            'history_window': list(range(150, 501, 25)), # 150-500 in steps of 25
            'rsi_length': list(range(10, 22)),         # 10-21
            'wt_channel': list(range(8, 15)),          # 8-14
            'wt_average': list(range(9, 16)),          # 9-15
            'cci_length': list(range(14, 31))          # 14-30
        }
        
        for _ in range(self.max_random_samples):
            params = {}
            for param, values in random_ranges.items():
                params[param] = random.choice(values)
            yield params
    
    def print_optimization_info(self):
        """Print detailed information about the optimization run"""
        approach, total = self.get_optimization_approach()
        
        print("\n" + "="*70)
        print("ULTRA-FAST LORENTZIAN OPTIMIZER")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Optimization Method: {approach.upper()}")
        
        if approach == "grid":
            print("\nGrid Search - Parameter Ranges:")
            print("-"*50)
            for param, values in self.param_grid.items():
                print(f"  {param:<20} : {values}")
        else:
            print("\nRandom Sampling - Parameter Ranges:")
            print("-"*50)
            print(f"  neighbors           : 3-20")
            print(f"  history_window      : 150-500 (step 25)")
            print(f"  rsi_length          : 10-21")
            print(f"  wt_channel          : 8-14")
            print(f"  wt_average          : 9-15")
            print(f"  cci_length          : 14-30")
        
        print("-"*50)
        print(f"Total combinations to test: {total}")
        print(f"Estimated time: ~{total * 3} seconds ({total * 3 / 60:.1f} minutes)")
        print("="*70 + "\n")
        
        return total
        
    def run_backtest(self, params, iteration, total):
        """Run a single backtest with given parameters"""
        # More concise output
        params_str = f"k={params['neighbors']}, win={params['history_window']}, rsi={params['rsi_length']}"
        print(f"[{iteration:2d}/{total}] {params_str} | ", end="", flush=True)
        
        try:
            strategy_params = {
                "symbols": [self.symbol],
                **params
            }
            
            result = LorentzianClassificationStrategy.backtest(
                datasource_class=PolygonDataBacktesting,
                start_datetime=self.start_date,
                end_datetime=self.end_date,
                benchmark_asset=Asset(self.symbol, Asset.AssetType.STOCK),
                buy_trading_fees=[TradingFee(percent_fee=0.001)],
                sell_trading_fees=[TradingFee(percent_fee=0.001)],
                quote_asset=Asset("USD", Asset.AssetType.FOREX),
                parameters=strategy_params,
                show_plot=False,
                save_tearsheet=False,
                show_tearsheet=False,
                quiet=True
            )
            
            # Extract return using the WORKING logic from simple optimizer
            total_return = 0.0
            final_value = 100000  # Default starting value
            
            # Try different ways to extract the return
            if hasattr(result, 'stats'):
                stats = result.stats
                if isinstance(stats, pd.DataFrame) and not stats.empty:
                    # Get the last row's cumulative return
                    if 'Cumulative Return' in stats.columns:
                        total_return = float(stats['Cumulative Return'].iloc[-1])
                    elif 'Total Return' in stats.columns:
                        total_return = float(stats['Total Return'].iloc[-1])
                        
                    # Also try to get final portfolio value
                    if 'Portfolio Value' in stats.columns:
                        final_value = float(stats['Portfolio Value'].iloc[-1])
            
            elif hasattr(result, 'stats_list') and result.stats_list:
                # Alternative way to get stats
                stats_dict = result.stats_list[0]
                total_return = stats_dict.get('total_return', 0)
                final_value = stats_dict.get('final_value', 100000)
                
            elif isinstance(result, dict):
                # If result is already a dictionary
                total_return = result.get('total_return', 0)
                final_value = result.get('final_value', 100000)
            
            # Calculate return from portfolio values if needed
            if total_return == 0 and final_value != 100000:
                total_return = (final_value - 100000) / 100000
            
            print(f"Return: {total_return:6.2%}, Final Value: ${final_value:,.2f}")
            return total_return
            
        except Exception as e:
            print(f"FAILED: {str(e)[:30]}")
            return -999
    
    def optimize(self):
        """Run ultra-fast optimization"""
        approach, total_combinations = self.get_optimization_approach()
        self.print_optimization_info()
        
        best_return = -999
        best_params = None
        all_results = []
        
        print("Starting optimization...\n")
        
        count = 0
        start_time = datetime.now()
        
        # Choose optimization method
        if approach == "grid":
            # Grid search with reduced parameters
            param_combinations = itertools.product(*self.param_grid.values())
            param_names = list(self.param_grid.keys())
            
            for param_values in param_combinations:
                count += 1
                params = dict(zip(param_names, param_values))
                
                ret = self.run_backtest(params, count, total_combinations)
                all_results.append((params, ret))
                
                if ret > best_return and ret != -999:
                    best_return = ret
                    best_params = params.copy()
                    print(f"    *** NEW BEST: {ret:.2%} ***")
        
        else:
            # Random sampling
            for params in self.generate_random_params():
                count += 1
                
                ret = self.run_backtest(params, count, total_combinations)
                all_results.append((params, ret))
                
                if ret > best_return and ret != -999:
                    best_return = ret
                    best_params = params.copy()
                    print(f"    *** NEW BEST: {ret:.2%} ***")
        
        # Results
        elapsed_min = (datetime.now() - start_time).total_seconds() / 60
        successful = [(p, r) for p, r in all_results if r != -999]
        successful.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Successful tests: {len(successful)}/{total_combinations}")
        print(f"Total time: {elapsed_min:.1f} minutes")
        print(f"Average time per test: {elapsed_min*60/total_combinations:.1f} seconds")
        
        print(f"\nTop 5 Results:")
        print("-"*50)
        for i, (p, r) in enumerate(successful[:5]):
            print(f"{i+1}. {r:6.2%} | k={p['neighbors']:2d}, win={p['history_window']:3d}, "
                  f"rsi={p['rsi_length']:2d}, wt_ch={p['wt_channel']:2d}, "
                  f"wt_avg={p['wt_average']:2d}, cci={p['cci_length']:2d}")
        
        if best_params:
            print(f"\n" + "="*40)
            print("BEST PARAMETERS:")
            print("="*40)
            for k, v in best_params.items():
                print(f"{k:<20} : {v}")
            print(f"Best Return: {best_return:.2%}")
            print("="*40)
        
        return best_params, best_return


# Example usage
if __name__ == "__main__":
    # Test with shorter date range for speed
    optimizer = UltraFastLorentzianOptimizer("TSLA", "2023-01-01", "2023-03-31")
    
    print("ULTRA-FAST OPTIMIZER - Choose your approach:")
    print("1. Grid search (6 combinations) - 2-3 minutes")
    print("2. Random sampling (50 combinations) - 10-15 minutes")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        optimizer.use_random_sampling = False
    else:
        optimizer.use_random_sampling = True
    
    best_params, best_return = optimizer.optimize() 