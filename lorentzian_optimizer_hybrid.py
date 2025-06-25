"""
Hybrid Ultra-Fast Lorentzian Optimizer
=====================================

This optimizer combines multiple speed techniques:
1. Parameter importance ranking (focus on most impactful parameters)
2. Early stopping (stop if no improvement after N iterations)
3. Shorter backtesting periods for initial screening
4. Multi-stage optimization (coarse -> fine)
5. Smart defaults based on trading literature

Designed to find good parameters in under 5-10 minutes.
"""

from datetime import datetime, timedelta
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy
import pandas as pd
import random
from itertools import product


class HybridLorentzianOptimizer:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Parameter importance ranking (based on ML and trading research)
        # 1 = most important, 3 = least important
        self.param_importance = {
            'neighbors': 1,        # K in k-NN is crucial for ML performance
            'history_window': 2,   # Important for data quality
            'rsi_length': 3,       # Standard is 14, less critical to optimize
            'wt_channel': 3,       # Wave Trend parameters are less critical
            'wt_average': 3,       # Wave Trend parameters are less critical
            'cci_length': 3        # CCI period is less critical
        }
        
        # Multi-stage parameter ranges
        self.stage1_params = {
            # Stage 1: Focus on most important parameters only
            'neighbors': [5, 8, 12, 15],      # Most critical parameter
            'history_window': [200, 300],     # Important for stability
            'rsi_length': [14],               # Use standard
            'wt_channel': [10],               # Use standard
            'wt_average': [11],               # Use standard
            'cci_length': [20]                # Use standard
        }
        
        self.stage2_params = {
            # Stage 2: Fine-tune around best Stage 1 result
            'neighbors': [],                  # Will be set based on Stage 1
            'history_window': [],             # Will be set based on Stage 1
            'rsi_length': [12, 14, 16],       # Now optimize secondary params
            'wt_channel': [8, 10, 12],
            'wt_average': [9, 11, 13],
            'cci_length': [18, 20, 22]
        }
        
        # Optimization settings
        self.early_stop_patience = 5  # Stop if no improvement for 5 iterations
        self.use_short_backtest = True  # Use shorter period for initial screening
        
        # Results tracking
        self.best_return = -999
        self.best_params = None
        self.no_improvement_count = 0
        self.all_results = []        
    def get_short_date_range(self):
        """Get shorter date range for faster initial screening"""
        if self.use_short_backtest:
            # Use last 2 months of the period for initial screening
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            start = end - timedelta(days=60)
            return start.strftime("%Y-%m-%d"), self.end_date
        return self.start_date, self.end_date
    
    def run_backtest(self, params, iteration, total, stage="", use_short=False):
        """Run a single backtest with given parameters"""
        start_date, end_date = self.get_short_date_range() if use_short else (self.start_date, self.end_date)
        
        # Compact output
        period_info = "(short)" if use_short else "(full)"
        print(f"[{iteration:2d}/{total}] {stage} k={params['neighbors']:2d}, "
              f"win={params['history_window']:3d} {period_info} | ", end="", flush=True)
        
        try:
            strategy_params = {
                "symbols": [self.symbol],
                **params
            }
            
            result = LorentzianClassificationStrategy.backtest(
                datasource_class=PolygonDataBacktesting,
                start_datetime=start_date,
                end_datetime=end_date,
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
    def check_early_stopping(self, current_return):
        """Check if we should stop early due to no improvement"""
        if current_return > self.best_return:
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            return self.no_improvement_count >= self.early_stop_patience
    
    def stage1_optimization(self):
        """Stage 1: Optimize most important parameters with short backtests"""
        print("\n" + "="*50)
        print("STAGE 1: Core Parameter Optimization (Short Period)")
        print("="*50)
        
        param_combinations = list(product(*self.stage1_params.values()))
        param_names = list(self.stage1_params.keys())
        total = len(param_combinations)
        
        print(f"Testing {total} combinations with shortened backtests...")
        
        best_stage1_return = -999
        best_stage1_params = None
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            ret = self.run_backtest(params, i+1, total, "S1", use_short=True)
            
            if ret > best_stage1_return and ret != -999:
                best_stage1_return = ret
                best_stage1_params = params.copy()
                print(f"    *** S1 BEST: {ret:.2%} ***")
            
            # Early stopping check
            if self.check_early_stopping(ret):
                print(f"\n    Early stopping at iteration {i+1} (no improvement for {self.early_stop_patience} tests)")
                break
        
        print(f"\nStage 1 Best: {best_stage1_return:.2%}")
        print(f"Best Stage 1 Parameters: {best_stage1_params}")
        
        return best_stage1_params, best_stage1_return    
    def stage2_optimization(self, best_stage1_params):
        """Stage 2: Fine-tune secondary parameters around Stage 1 best"""
        print("\n" + "="*50)
        print("STAGE 2: Fine-tuning Secondary Parameters (Full Period)")
        print("="*50)
        
        # Set Stage 2 parameter ranges around Stage 1 best
        k_best = best_stage1_params['neighbors']
        win_best = best_stage1_params['history_window']
        
        # Create narrow ranges around best values
        self.stage2_params['neighbors'] = [max(3, k_best-2), k_best, min(20, k_best+2)]
        self.stage2_params['history_window'] = [max(150, win_best-50), win_best, min(500, win_best+50)]
        
        param_combinations = list(product(*self.stage2_params.values()))
        param_names = list(self.stage2_params.keys())
        total = len(param_combinations)
        
        print(f"Testing {total} combinations with full backtests...")
        print(f"Neighbors range: {self.stage2_params['neighbors']}")
        print(f"History window range: {self.stage2_params['history_window']}")
        
        self.no_improvement_count = 0  # Reset early stopping counter
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            ret = self.run_backtest(params, i+1, total, "S2", use_short=False)
            self.all_results.append((params.copy(), ret))
            
            if ret > self.best_return and ret != -999:
                self.best_return = ret
                self.best_params = params.copy()
                print(f"    *** S2 BEST: {ret:.2%} ***")
            
            # Early stopping check
            if self.check_early_stopping(ret):
                print(f"\n    Early stopping at iteration {i+1} (no improvement for {self.early_stop_patience} tests)")
                break
        
        return self.best_params, self.best_return    
    def optimize(self):
        """Run hybrid two-stage optimization"""
        print("\n" + "="*70)
        print("HYBRID LORENTZIAN OPTIMIZER")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Strategy: Two-stage optimization with early stopping")
        print(f"Early stop patience: {self.early_stop_patience} iterations")
        
        start_time = datetime.now()
        
        # Stage 1: Core parameters with short backtests
        best_stage1_params, best_stage1_return = self.stage1_optimization()
        
        if best_stage1_params is None:
            print("Stage 1 failed - no successful backtests")
            return None, None
        
        # Stage 2: Fine-tune with full backtests
        best_params, best_return = self.stage2_optimization(best_stage1_params)
        
        # Final results
        elapsed_min = (datetime.now() - start_time).total_seconds() / 60
        successful = [(p, r) for p, r in self.all_results if r != -999]
        successful.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n" + "="*70)
        print("HYBRID OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Total time: {elapsed_min:.1f} minutes")
        print(f"Stage 2 successful tests: {len(successful)}")
        
        if successful:
            print(f"\nTop 3 Stage 2 Results:")
            print("-"*50)
            for i, (p, r) in enumerate(successful[:3]):
                print(f"{i+1}. {r:6.2%} | k={p['neighbors']:2d}, win={p['history_window']:3d}, "
                      f"rsi={p['rsi_length']:2d}, wt_ch={p['wt_channel']:2d}, "
                      f"wt_avg={p['wt_average']:2d}, cci={p['cci_length']:2d}")
        
        if best_params:
            print(f"\n" + "="*40)
            print("FINAL BEST PARAMETERS:")
            print("="*40)
            for k, v in best_params.items():
                print(f"{k:<20} : {v}")
            print(f"Best Return: {best_return:.2%}")
            print("="*40)
        
        return best_params, best_return

# Quick test optimizer for ultra-fast results
class QuickTestOptimizer:
    """Ultra-quick optimizer for immediate feedback (5-10 tests)"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
    def optimize(self):
        print("QUICK TEST OPTIMIZER - Testing 8 smart combinations...")
        
        # Hand-picked parameter combinations based on trading research
        smart_combinations = [
            {'neighbors': 5, 'history_window': 200, 'rsi_length': 14, 'wt_channel': 10, 'wt_average': 11, 'cci_length': 20},
            {'neighbors': 8, 'history_window': 250, 'rsi_length': 14, 'wt_channel': 10, 'wt_average': 11, 'cci_length': 20},
            {'neighbors': 12, 'history_window': 300, 'rsi_length': 14, 'wt_channel': 10, 'wt_average': 11, 'cci_length': 20},
            {'neighbors': 8, 'history_window': 200, 'rsi_length': 12, 'wt_channel': 8, 'wt_average': 9, 'cci_length': 18},
            {'neighbors': 8, 'history_window': 300, 'rsi_length': 16, 'wt_channel': 12, 'wt_average': 13, 'cci_length': 22},
            {'neighbors': 15, 'history_window': 350, 'rsi_length': 14, 'wt_channel': 10, 'wt_average': 11, 'cci_length': 20},
            {'neighbors': 6, 'history_window': 180, 'rsi_length': 14, 'wt_channel': 10, 'wt_average': 11, 'cci_length': 20},
            {'neighbors': 10, 'history_window': 400, 'rsi_length': 14, 'wt_channel': 10, 'wt_average': 11, 'cci_length': 20}
        ]
        
        best_return = -999
        best_params = None
        
        for i, params in enumerate(smart_combinations):
            print(f"\n[{i+1}/8] k={params['neighbors']}, win={params['history_window']}")
            
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
        
        print(f"\nQuick test complete - Best return: {best_return:.2%}")
        return best_params, best_return


# Example usage
if __name__ == "__main__":
    print("Choose optimization method:")
    print("1. Quick test (8 combinations, ~2-3 minutes)")
    print("2. Hybrid optimization (20-30 combinations, ~5-10 minutes)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        optimizer = QuickTestOptimizer("TSLA", "2023-01-01", "2023-03-31")
        best_params, best_return = optimizer.optimize()
    else:
        optimizer = HybridLorentzianOptimizer("TSLA", "2023-01-01", "2023-03-31")
        best_params, best_return = optimizer.optimize()
    
    if best_params:
        print(f"\nTo use these parameters:")
        print(f"parameters = {best_params}")