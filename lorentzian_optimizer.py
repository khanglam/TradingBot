from datetime import datetime
import itertools
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy


class LorentzianOptimizer:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Define parameter ranges
        self.param_ranges = {
            'neighbors': range(3, 21),  # 3-20
            'history_window': range(100, 501, 50),  # 100-500, step 50
            'rsi_length': range(10, 22),  # 10-21
            'wt_channel': range(5, 16),  # 5-15
            'wt_average': range(8, 16),  # 8-15
            'cci_length': range(14, 26),  # 14-25
        }
        
    def run_backtest(self, params):
        """Run a single backtest with given parameters"""
        try:
            # Create parameter dictionary for the strategy
            strategy_params = {
                "symbols": [self.symbol],
                "neighbors": params['neighbors'],
                "history_window": params['history_window'],
                "rsi_length": params['rsi_length'],
                "wt_channel": params['wt_channel'],
                "wt_average": params['wt_average'],
                "cci_length": params['cci_length']
            }
            
            # Run backtest
            trading_fee = TradingFee(percent_fee=0.001)
            result = LorentzianClassificationStrategy.backtest(
                datasource_class=PolygonDataBacktesting,
                start_datetime=self.start_date,
                end_datetime=self.end_date,
                benchmark_asset=Asset(self.symbol, Asset.AssetType.STOCK),
                buy_trading_fees=[trading_fee],
                sell_trading_fees=[trading_fee],
                quote_asset=Asset("USD", Asset.AssetType.FOREX),
                parameters=strategy_params,
                show_plot=False,  # Don't show plots during optimization
                save_tearsheet=False,  # Don't save tearsheets
                show_tearsheet=False
            )
            
            # Extract total return from results
            stats = result['stats'] if isinstance(result, dict) else result
            total_return = float(stats.get('total_return', 0))
            
            return total_return
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
            return -999  # Return very negative value for failed runs
    
    def optimize(self):
        """Run optimization and return best parameters"""
        best_return = -999
        best_params = None
        results = []
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            self.param_ranges['neighbors'],
            self.param_ranges['history_window'],
            self.param_ranges['rsi_length'],
            self.param_ranges['wt_channel'],
            self.param_ranges['wt_average'],
            self.param_ranges['cci_length']
        ))
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations for {self.symbol}...")
        
        # Test each combination
        for i, (n, hw, rsi, wtc, wta, cci) in enumerate(param_combinations):
            params = {
                'neighbors': n,
                'history_window': hw,
                'rsi_length': rsi,
                'wt_channel': wtc,
                'wt_average': wta,
                'cci_length': cci
            }
            
            # Show progress
            if i % 10 == 0:
                print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            # Run backtest
            total_return = self.run_backtest(params)
            
            # Track results
            results.append((params, total_return))
            
            # Update best if needed
            if total_return > best_return:
                best_return = total_return
                best_params = params
                print(f"New best! Return: {best_return:.2%} with params: {best_params}")
        
        # Sort results and show top 5
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 parameter combinations:")
        for i, (params, ret) in enumerate(results[:5]):
            print(f"{i+1}. Return: {ret:.2%} | {params}")
        
        return best_params, best_return


# Example usage
if __name__ == "__main__":
    optimizer = LorentzianOptimizer("TSLA", "2023-01-01", "2024-01-01")
    best_params, best_return = optimizer.optimize()
    print(f"\nBest params: {best_params}")
    print(f"Best return: {best_return:.2%}") 