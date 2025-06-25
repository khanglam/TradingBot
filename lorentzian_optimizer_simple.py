from datetime import datetime
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy
import pandas as pd


class LorentzianOptimizer:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Reduced parameter ranges for faster testing
        self.param_grid = {
            'neighbors': [5, 8, 12, 15],
            'history_window': [200, 300, 400],
            'rsi_length': [10, 14, 18],
            'wt_channel': [8, 10, 12],
            'wt_average': [9, 11, 13],
            'cci_length': [14, 20, 25]
        }
        
    def print_optimization_info(self):
        """Print detailed information about the optimization run"""
        print("\n" + "="*70)
        print("LORENTZIAN CLASSIFICATION STRATEGY OPTIMIZER")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print("\nParameter Ranges:")
        print("-"*50)
        for param, values in self.param_grid.items():
            print(f"  {param:<20} : {values}")
        
        # Calculate total combinations
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        
        print("-"*50)
        print(f"Total combinations to test: {total}")
        print(f"Estimated time: ~{total * 5} seconds ({total * 5 / 60:.1f} minutes)")
        print("="*70 + "\n")
        
        return total
        
    def run_backtest(self, params, iteration, total):
        """Run a single backtest with given parameters"""
        # Print current test info
        print(f"\n[{iteration}/{total}] Testing parameters:")
        for k, v in params.items():
            print(f"    {k:<20} : {v}")
        
        try:
            # Create parameter dictionary
            strategy_params = {
                "symbols": [self.symbol],
                **params  # Unpack all parameters
            }
            
            # Run backtest
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
                quiet=True  # Suppress output
            )
            
            # Extract metrics from the result
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
                
            print(f"    Result: Return = {total_return:.2%}, Final Value = ${final_value:,.2f}")
            
            return total_return
            
        except Exception as e:
            print(f"    FAILED: {str(e)}")
            return -999
    
    def optimize(self):
        """Run grid search optimization"""
        # Print optimization info
        total_combinations = self.print_optimization_info()
        
        best_return = -999
        best_params = None
        all_results = []
        
        print("\nStarting optimization...\n")
        print("-"*70)
        
        count = 0
        start_time = datetime.now()
        
        # Grid search through all combinations
        for n in self.param_grid['neighbors']:
            for hw in self.param_grid['history_window']:
                for rsi in self.param_grid['rsi_length']:
                    for wtc in self.param_grid['wt_channel']:
                        for wta in self.param_grid['wt_average']:
                            for cci in self.param_grid['cci_length']:
                                count += 1
                                params = {
                                    'neighbors': n,
                                    'history_window': hw,
                                    'rsi_length': rsi,
                                    'wt_channel': wtc,
                                    'wt_average': wta,
                                    'cci_length': cci
                                }
                                
                                # Run backtest
                                ret = self.run_backtest(params, count, total_combinations)
                                all_results.append((params, ret))
                                
                                # Update best
                                if ret > best_return and ret != -999:
                                    best_return = ret
                                    best_params = params.copy()
                                    print(f"\n*** NEW BEST! Return: {ret:.2%} ***\n")
                                
                                # Progress estimate
                                elapsed = (datetime.now() - start_time).total_seconds()
                                if count > 0:
                                    eta = (elapsed / count) * (total_combinations - count)
                                    print(f"    Progress: {count/total_combinations*100:.1f}% | "
                                          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        
        # Final results
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        
        # Sort and show top 5
        all_results = [(p, r) for p, r in all_results if r != -999]  # Filter out failed runs
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTotal successful backtests: {len(all_results)} out of {total_combinations}")
        print(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
        
        print("\nTop 5 Results:")
        print("-"*70)
        for i, (p, r) in enumerate(all_results[:5]):
            print(f"\n{i+1}. Return: {r:.2%}")
            for k, v in p.items():
                print(f"   {k:<20} : {v}")
        
        print("\n" + "="*70)
        print("BEST PARAMETERS FOUND:")
        print("="*70)
        if best_params:
            for k, v in best_params.items():
                print(f"{k:<20} : {v}")
            print(f"\nBest Return: {best_return:.2%}")
        else:
            print("No successful optimizations found!")
        print("="*70)
        
        return best_params, best_return


# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = LorentzianOptimizer("TSLA", "2023-01-01", "2023-06-30")
    
    # Run optimization
    best_params, best_return = optimizer.optimize()
    
    # Save results
    if best_params:
        print(f"\nTo use these parameters, update your strategy with:")
        print(f"parameters = {best_params}") 