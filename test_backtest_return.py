from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import PolygonDataBacktesting
from LorentzianClassificationStrategy import LorentzianClassificationStrategy
import pprint

# Test parameters
test_params = {
    "symbols": ["TSLA"],
    "neighbors": 8,
    "history_window": 300,
    "rsi_length": 14,
    "wt_channel": 10,
    "wt_average": 11,
    "cci_length": 20
}

print("Running test backtest to check return calculation...")
print(f"Parameters: {test_params}")

# Run backtest
result = LorentzianClassificationStrategy.backtest(
    datasource_class=PolygonDataBacktesting,
    start_datetime="2023-01-01",
    end_datetime="2023-06-30",
    benchmark_asset=Asset("TSLA", Asset.AssetType.STOCK),
    buy_trading_fees=[TradingFee(percent_fee=0.001)],
    sell_trading_fees=[TradingFee(percent_fee=0.001)],
    quote_asset=Asset("USD", Asset.AssetType.FOREX),
    parameters=test_params,
    show_plot=False,
    save_tearsheet=False,
    show_tearsheet=False
)

print("\n" + "="*50)
print("RESULT INSPECTION:")
print("="*50)

# Check what type result is
print(f"Result type: {type(result)}")
print(f"Result attributes: {dir(result)}")

# Try to extract return in different ways
print("\nTrying to extract return value...")

if hasattr(result, 'stats'):
    print(f"\nFound 'stats' attribute")
    print(f"Stats type: {type(result.stats)}")
    if hasattr(result.stats, 'columns'):
        print(f"Stats columns: {list(result.stats.columns)}")
        print(f"\nFirst few rows of stats:")
        print(result.stats.head())
        print(f"\nLast few rows of stats:")
        print(result.stats.tail())

if hasattr(result, 'stats_list'):
    print(f"\nFound 'stats_list' attribute")
    if result.stats_list:
        print(f"First stats dict keys: {list(result.stats_list[0].keys())}")
        print(f"\nStats values:")
        pprint.pprint(result.stats_list[0])

if hasattr(result, 'results'):
    print(f"\nFound 'results' attribute")
    print(f"Results type: {type(result.results)}")

# Try to get the actual return value
print("\n" + "="*50)
print("EXTRACTED VALUES:")
print("="*50)

if hasattr(result, 'stats') and hasattr(result.stats, 'iloc'):
    # Try to get from DataFrame
    try:
        if 'Total Return' in result.stats.columns:
            total_return = result.stats['Total Return'].iloc[-1]
            print(f"Total Return from stats: {total_return:.2%}")
        if 'Cumulative Returns' in result.stats.columns:
            cum_return = result.stats['Cumulative Returns'].iloc[-1]
            print(f"Cumulative Returns from stats: {cum_return:.2%}")
        if 'Portfolio Value' in result.stats.columns:
            final_value = result.stats['Portfolio Value'].iloc[-1]
            initial_value = result.stats['Portfolio Value'].iloc[0]
            calc_return = (final_value - initial_value) / initial_value
            print(f"Portfolio Value: Initial=${initial_value:,.2f}, Final=${final_value:,.2f}")
            print(f"Calculated Return: {calc_return:.2%}")
    except Exception as e:
        print(f"Error extracting from stats DataFrame: {e}")

if hasattr(result, 'stats_list') and result.stats_list:
    try:
        stats_dict = result.stats_list[0]
        print(f"\nFrom stats_list:")
        for key in ['total_return', 'total_return_pct', 'cumulative_return', 'final_value']:
            if key in stats_dict:
                print(f"  {key}: {stats_dict[key]}")
    except Exception as e:
        print(f"Error extracting from stats_list: {e}") 