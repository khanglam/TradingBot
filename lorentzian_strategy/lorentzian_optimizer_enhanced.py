from datetime import datetime
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy
import pandas as pd
import json


class LorentzianOptimizer:
    def __init__(self, symbol, start_date, end_date, initial_cash=100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        
        # Parameter ranges
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
        print("\n" + "="*80)
        print(" " * 15 + "LORENTZIAN CLASSIFICATION STRATEGY OPTIMIZER")
        print("="*80)
        print(f"{'Symbol:':<20} {self.symbol}")
        print(f"{'Date Range:':<20} {self.start_date} to {self.end_date}")
        print(f"{'Initial Cash:':<20} ${self.initial_cash:,.2f}")
        
        print("\n" + "PARAMETER SEARCH SPACE:")
        print("-"*80)
        print(f"{'Parameter':<20} {'Values':<30} {'Count':<10}")
        print("-"*80)
        
        total = 1
        for param, values in self.param_grid.items():
            total *= len(values)
            values_str = str(values) if len(str(values)) <= 30 else str(values)[:27] + "..."
            print(f"{param:<20} {values_str:<30} {len(values):<10}")
        
        print("-"*80)
        print(f"{'TOTAL COMBINATIONS:':<20} {total}")
        
        # Estimate time
        avg_seconds_per_test = 5  # Rough estimate
        total_seconds = total * avg_seconds_per_test
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        print(f"{'ESTIMATED TIME:':<20} {hours}h {minutes}m {seconds}s")
        print("="*80 + "\n")
        
        return total
        
    def extract_return_from_result(self, result):
        """Extract the total return from various possible result formats"""
        try:
            # Method 1: Direct attribute access
            if hasattr(result, 'total_return'):
                return float(result.total_return)
            
            # Method 2: From results dictionary
            if hasattr(result, 'results') and isinstance(result.results, dict):
                # Check for strategy-specific results
                for strategy_name, strategy_result in result.results.items():
                    if hasattr(strategy_result, 'total_return'):
                        return float(strategy_result.total_return)
                    if isinstance(strategy_result, dict) and 'total_return' in strategy_result:
                        return float(strategy_result['total_return'])
            
            # Method 3: From stats DataFrame
            if hasattr(result, 'stats') and isinstance(result.stats, pd.DataFrame):
                df = result.stats
                if not df.empty:
                    # Try different column names
                    for col in ['total_return', 'Total Return', 'Cumulative Returns', 'cumulative_returns']:
                        if col in df.columns:
                            return float(df[col].iloc[-1])
                    
                    # Calculate from portfolio value
                    if 'Portfolio Value' in df.columns:
                        final_value = float(df['Portfolio Value'].iloc[-1])
                        initial_value = float(df['Portfolio Value'].iloc[0])
                        if initial_value > 0:
                            return (final_value - initial_value) / initial_value
            
            # Method 4: From stats_list
            if hasattr(result, 'stats_list') and result.stats_list:
                stats_dict = result.stats_list[0]
                if 'total_return' in stats_dict:
                    return float(stats_dict['total_return'])
                if 'total_return_pct' in stats_dict:
                    return float(stats_dict['total_return_pct']) / 100
                
                # Calculate from portfolio values
                if 'final_value' in stats_dict and 'initial_value' in stats_dict:
                    initial = float(stats_dict['initial_value'])
                    final = float(stats_dict['final_value'])
                    if initial > 0:
                        return (final - initial) / initial
            
            # Method 5: Look for any numeric return value
            if isinstance(result, dict):
                for key in ['total_return', 'return', 'total_return_pct', 'cumulative_return']:
                    if key in result:
                        val = float(result[key])
                        # If it looks like a percentage (> 1), convert to decimal
                        if key.endswith('_pct') and abs(val) > 1:
                            val = val / 100
                        return val
            
            print("WARNING: Could not extract return value from result")
            return 0.0
            
        except Exception as e:
            print(f"ERROR extracting return: {e}")
            return 0.0
        
    def run_backtest(self, params, iteration, total):
        """Run a single backtest with given parameters"""
        # Format parameter display
        param_str = " | ".join([f"{k}={v}" for k, v in params.items()])
        print(f"\n[{iteration:3d}/{total:3d}] Testing: {param_str}")
        
        try:
            # Create parameter dictionary
            strategy_params = {
                "symbols": [self.symbol],
                **params
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
                budget=self.initial_cash,
                show_plot=False,
                save_tearsheet=False,
                show_tearsheet=False,
                quiet=True
            )
            
            # Extract return
            total_return = self.extract_return_from_result(result)
            
            # Calculate additional metrics if possible
            sharpe_ratio = None
            max_drawdown = None
            
            if hasattr(result, 'stats') and isinstance(result.stats, pd.DataFrame):
                df = result.stats
                if 'Sharpe Ratio' in df.columns:
                    sharpe_ratio = float(df['Sharpe Ratio'].iloc[-1])
                if 'Max Drawdown' in df.columns:
                    max_drawdown = float(df['Max Drawdown'].iloc[-1])
            
            # Print results
            print(f"{'':>13} ✓ Return: {total_return:>7.2%}", end="")
            if sharpe_ratio is not None:
                print(f" | Sharpe: {sharpe_ratio:>5.2f}", end="")
            if max_drawdown is not None:
                print(f" | MaxDD: {max_drawdown:>6.2%}", end="")
            print()
            
            return total_return, sharpe_ratio, max_drawdown
            
        except Exception as e:
            print(f"{'':>13} ✗ FAILED: {str(e)[:60]}...")
            return -999, None, None
    
    def optimize(self):
        """Run grid search optimization"""
        # Print optimization info
        total_combinations = self.print_optimization_info()
        
        best_return = -999
        best_params = None
        best_metrics = {}
        all_results = []
        
        print("STARTING OPTIMIZATION...")
        print("-"*80)
        
        count = 0
        start_time = datetime.now()
        successful_runs = 0
        
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
                                ret, sharpe, max_dd = self.run_backtest(params, count, total_combinations)
                                
                                if ret != -999:
                                    successful_runs += 1
                                    result_dict = {
                                        'params': params.copy(),
                                        'return': ret,
                                        'sharpe': sharpe,
                                        'max_drawdown': max_dd
                                    }
                                    all_results.append(result_dict)
                                    
                                    # Update best
                                    if ret > best_return:
                                        best_return = ret
                                        best_params = params.copy()
                                        best_metrics = result_dict
                                        print(f"\n{'':>13} ★ NEW BEST! Return: {ret:.2%} ★\n")
                                
                                # Progress update
                                if count % 5 == 0 or count == total_combinations:
                                    elapsed = (datetime.now() - start_time).total_seconds()
                                    if count > 0:
                                        avg_time = elapsed / count
                                        eta = avg_time * (total_combinations - count)
                                        success_rate = (successful_runs / count) * 100
                                        print(f"\n{'':>13} Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) | "
                                              f"Success rate: {success_rate:.0f}% | "
                                              f"ETA: {eta/60:.1f}min")
        
        # Final results
        print("\n" + "="*80)
        print(" " * 25 + "OPTIMIZATION COMPLETE")
        print("="*80)
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"{'Total time:':<25} {total_time/60:.1f} minutes")
        print(f"{'Total backtests:':<25} {total_combinations}")
        print(f"{'Successful backtests:':<25} {successful_runs} ({successful_runs/total_combinations*100:.1f}%)")
        print(f"{'Failed backtests:':<25} {total_combinations - successful_runs}")
        
        if all_results:
            # Sort by return
            all_results.sort(key=lambda x: x['return'], reverse=True)
            
            print("\n" + "TOP 10 RESULTS:")
            print("-"*80)
            print(f"{'Rank':<6} {'Return':<10} {'Sharpe':<10} {'MaxDD':<10} Parameters")
            print("-"*80)
            
            for i, result in enumerate(all_results[:10]):
                rank = f"#{i+1}"
                ret_str = f"{result['return']:.2%}"
                sharpe_str = f"{result['sharpe']:.2f}" if result['sharpe'] else "N/A"
                dd_str = f"{result['max_drawdown']:.2%}" if result['max_drawdown'] else "N/A"
                
                print(f"{rank:<6} {ret_str:<10} {sharpe_str:<10} {dd_str:<10}", end=" ")
                param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
                print(param_str)
            
            # Best parameters summary
            print("\n" + "="*80)
            print(" " * 25 + "BEST PARAMETERS FOUND")
            print("="*80)
            
            if best_params:
                print(f"\n{'Return:':<25} {best_return:.2%}")
                if best_metrics.get('sharpe'):
                    print(f"{'Sharpe Ratio:':<25} {best_metrics['sharpe']:.2f}")
                if best_metrics.get('max_drawdown'):
                    print(f"{'Max Drawdown:':<25} {best_metrics['max_drawdown']:.2%}")
                
                print(f"\n{'Parameters:':<25}")
                for k, v in best_params.items():
                    print(f"  {k:<23} = {v}")
                
                # Save results to file
                results_file = f"optimization_results_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        'symbol': self.symbol,
                        'date_range': f"{self.start_date} to {self.end_date}",
                        'best_parameters': best_params,
                        'best_return': best_return,
                        'all_results': all_results[:20]  # Save top 20
                    }, f, indent=2)
                
                print(f"\n{'Results saved to:':<25} {results_file}")
                
                # Show how to use the parameters
                print("\n" + "TO USE THESE PARAMETERS:")
                print("-"*80)
                print("Update your LorentzianClassificationStrategy with:")
                print(f"\nparameters = {{")
                print(f'    "symbols": ["{self.symbol}"],')
                for k, v in best_params.items():
                    print(f'    "{k}": {v},')
                print("}")
        else:
            print("\nNo successful optimizations found!")
        
        print("="*80)
        
        return best_params, best_return


# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = LorentzianOptimizer(
        symbol="TSLA",
        start_date="2023-01-01",
        end_date="2023-06-30",
        initial_cash=100000
    )
    
    # Run optimization
    best_params, best_return = optimizer.optimize() 