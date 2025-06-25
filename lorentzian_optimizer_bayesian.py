"""
Bayesian Optimization for Lorentzian Strategy
=============================================

This optimizer uses Bayesian optimization to intelligently search the parameter space.
Instead of testing all combinations, it learns from previous results to predict
which parameter combinations are most likely to perform well.

Benefits:
- Requires only 20-50 evaluations instead of 900+
- Finds optimal parameters faster
- Provides uncertainty estimates
- Can handle continuous parameter spaces

Requirements: pip install scikit-optimize
"""

from datetime import datetime
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy
import pandas as pd

try:
    from skopt import gp_minimize
    from skopt.space import Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("WARNING: scikit-optimize not installed. Install with: pip install scikit-optimize")


class BayesianLorentzianOptimizer:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Define parameter search space for Bayesian optimization
        self.dimensions = [
            Integer(3, 20, name='neighbors'),           # k in k-NN
            Integer(150, 500, name='history_window'),   # Historical data window
            Integer(10, 21, name='rsi_length'),         # RSI period
            Integer(8, 14, name='wt_channel'),          # Wave Trend channel
            Integer(9, 15, name='wt_average'),          # Wave Trend average
            Integer(14, 30, name='cci_length')          # CCI period
        ]
        
        # Optimization settings
        self.n_calls = 30  # Number of evaluations (much less than 972!)
        self.n_initial_points = 10  # Random points to start with
        
        # Track results
        self.iteration = 0
        self.best_return = -999
        self.best_params = None
        self.all_results = []
        
    def print_optimization_info(self):
        """Print optimization information"""
        print("\n" + "="*70)
        print("BAYESIAN LORENTZIAN OPTIMIZER")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Method: Gaussian Process Bayesian Optimization")
        
        print("\nParameter Search Space:")
        print("-"*50)
        for dim in self.dimensions:
            print(f"  {dim.name:<20} : {dim.low} to {dim.high}")
        
        print("-"*50)
        print(f"Total evaluations: {self.n_calls}")
        print(f"Initial random points: {self.n_initial_points}")
        print(f"Estimated time: ~{self.n_calls * 3} seconds ({self.n_calls * 3 / 60:.1f} minutes)")
        print("="*70 + "\n")
        
    def objective_function(self, params):
        """
        Objective function for Bayesian optimization.
        Takes parameter list and returns NEGATIVE return (for minimization).
        """
        self.iteration += 1
        
        # Convert parameter list to dictionary
        param_dict = {
            'neighbors': params[0],
            'history_window': params[1], 
            'rsi_length': params[2],
            'wt_channel': params[3],
            'wt_average': params[4],
            'cci_length': params[5]
        }
        
        # Compact output
        print(f"[{self.iteration:2d}/{self.n_calls}] k={param_dict['neighbors']:2d}, "
              f"win={param_dict['history_window']:3d}, rsi={param_dict['rsi_length']:2d} | ", 
              end="", flush=True)
        
        try:
            strategy_params = {
                "symbols": [self.symbol],
                **param_dict
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
            
            # Store results
            self.all_results.append((param_dict.copy(), total_return))
            
            # Track best
            if total_return > self.best_return:
                self.best_return = total_return
                self.best_params = param_dict.copy()
                print(f"    *** NEW BEST: {total_return:.2%} ***")
            
            # Return NEGATIVE return for minimization
            return -total_return
            
        except Exception as e:
            print(f"FAILED: {str(e)[:30]}")
            return 999  # Large positive value for failed runs
    
    def optimize(self):
        """Run Bayesian optimization"""
        if not BAYESIAN_AVAILABLE:
            print("ERROR: scikit-optimize is required for Bayesian optimization")
            print("Install with: pip install scikit-optimize")
            return None, None
            
        self.print_optimization_info()
        
        print("Starting Bayesian optimization...\n")
        start_time = datetime.now()
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=self.dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=42,
            acq_func='EI',  # Expected Improvement
            verbose=False
        )
        
        # Results
        elapsed_min = (datetime.now() - start_time).total_seconds() / 60
        successful = [(p, r) for p, r in self.all_results if r != -999]
        successful.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n" + "="*70)
        print("BAYESIAN OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Successful tests: {len(successful)}/{self.n_calls}")
        print(f"Total time: {elapsed_min:.1f} minutes")
        print(f"Average time per test: {elapsed_min*60/self.n_calls:.1f} seconds")
        print(f"Best return found: {-result.fun:.2%}")
        
        print(f"\nTop 5 Results:")
        print("-"*50)
        for i, (p, r) in enumerate(successful[:5]):
            print(f"{i+1}. {r:6.2%} | k={p['neighbors']:2d}, win={p['history_window']:3d}, "
                  f"rsi={p['rsi_length']:2d}, wt_ch={p['wt_channel']:2d}, "
                  f"wt_avg={p['wt_average']:2d}, cci={p['cci_length']:2d}")
        
        if self.best_params:
            print(f"\n" + "="*40)
            print("BEST PARAMETERS:")
            print("="*40)
            for k, v in self.best_params.items():
                print(f"{k:<20} : {v}")
            print(f"Best Return: {self.best_return:.2%}")
            print("="*40)
            
            # Bayesian optimization insights
            print(f"\nBayesian Optimization Insights:")
            print(f"Convergence: Found optimum in {self.iteration} evaluations")
            print(f"Efficiency: {((972 - self.n_calls) / 972 * 100):.1f}% fewer tests than full grid search")
        
        return self.best_params, self.best_return


class FallbackRandomOptimizer:
    """Fallback optimizer if Bayesian optimization is not available"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.n_samples = 30
        
    def optimize(self):
        print("Using fallback random optimizer...")
        
        import random
        
        best_return = -999
        best_params = None
        
        for i in range(self.n_samples):
            params = {
                'neighbors': random.randint(3, 20),
                'history_window': random.randint(150, 500),
                'rsi_length': random.randint(10, 21),
                'wt_channel': random.randint(8, 14),
                'wt_average': random.randint(9, 15),
                'cci_length': random.randint(14, 30)
            }
            
            print(f"[{i+1}/{self.n_samples}] Testing random parameters...")
            
            # Same backtest logic as Bayesian optimizer
            try:
                strategy_params = {"symbols": [self.symbol], **params}
                
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
                
                total_return = 0.0
                if hasattr(result, 'stats') and isinstance(result.stats, pd.DataFrame):
                    stats = result.stats
                    if not stats.empty and 'Cumulative Return' in stats.columns:
                        total_return = float(stats['Cumulative Return'].iloc[-1])
                
                print(f"Return: {total_return:.2%}")
                
                if total_return > best_return:
                    best_return = total_return
                    best_params = params.copy()
                    print(f"*** NEW BEST: {total_return:.2%} ***")
                    
            except Exception as e:
                print(f"Failed: {e}")
        
        return best_params, best_return


# Example usage
if __name__ == "__main__":
    # Shorter test period for speed
    optimizer = BayesianLorentzianOptimizer("TSLA", "2023-01-01", "2023-03-31")
    
    if BAYESIAN_AVAILABLE:
        print("Using Bayesian optimization (30 evaluations)")
        best_params, best_return = optimizer.optimize()
    else:
        print("Bayesian optimization not available, using random search")
        fallback = FallbackRandomOptimizer("TSLA", "2023-01-01", "2023-03-31")
        best_params, best_return = fallback.optimize()
    
    if best_params:
        print(f"\nTo use these parameters in your strategy:")
        print(f"parameters = {best_params}") 