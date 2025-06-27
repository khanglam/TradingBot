"""
Data Source Investigation Tool - COMPREHENSIVE ANALYSIS
======================================================

This script investigates data differences between:
1. Direct Polygon API (used by optimization scripts)  
2. Lumibot's get_historical_prices during ACTUAL STRATEGY EXECUTION (cache behavior)

Key insights from previous analysis:
- Lumibot uses a cache system that may contain years of data
- get_historical_prices(length=N) returns the LAST N rows from cache, not just backtest window
- This explains why optimization (direct API) != strategy performance (cache behavior)

This script simulates EXACTLY how AdvancedLorentzianStrategy retrieves data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
import shutil
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_polygon_direct(symbol, start_date, end_date):
    """Download data using direct Polygon API (same as optimization scripts)"""
    try:
        from test_parameters import download_real_data
        print(f"📥 Downloading data using direct Polygon API...")
        print(f"   Requesting exact backtest window: {start_date} to {end_date}")
        
        # Use the same function as optimization scripts
        data = download_real_data(symbol=symbol, start_date=start_date, end_date=end_date, timeframe='day')
        
        if data is not None and not data.empty:
            # Convert to match Lumibot format
            data.index = pd.to_datetime(data.index)
            print(f"✅ Direct Polygon API: {len(data)} rows from {data.index[0].date()} to {data.index[-1].date()}")
            
            # Check Polygon Free API limitation
            today = datetime.now().date()
            data_start = data.index[0].date()
            days_back = (today - data_start).days
            
            if days_back > 730:  # ~2 years
                print(f"⚠️  Note: Data goes back {days_back} days (~{days_back/365:.1f} years)")
                print(f"   Free Polygon API typically limits to ~2 years of data")
            
            return data
        else:
            print("❌ Failed to download data from direct Polygon API")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading direct Polygon data: {str(e)}")
        return None

class TestStrategy:
    """Minimal strategy that mimics AdvancedLorentzianStrategy's data retrieval"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.retrieved_data = None
        
    def get_historical_prices(self, asset, length, timeframe):
        """Simulate Lumibot's get_historical_prices method"""
        from lumibot.backtesting import PolygonDataBacktesting
        from lumibot.entities import Asset
        
        try:
            # Create datasource (this will use Lumibot's cache behavior)
            start_dt = datetime.strptime(os.getenv('BACKTESTING_START'), '%Y-%m-%d')
            end_dt = datetime.strptime(os.getenv('BACKTESTING_END'), '%Y-%m-%d')
            
            datasource = PolygonDataBacktesting(
                datetime_start=start_dt,
                datetime_end=end_dt,
                api_key=os.getenv('POLYGON_API_KEY')
            )
            
            # This is the EXACT call that AdvancedLorentzianStrategy makes
            bars = datasource.get_historical_prices(asset, length, timeframe)
            
            if bars and len(bars.df) > 0:
                return bars
            else:
                print(f"❌ No data returned from get_historical_prices")
                return None
                
        except Exception as e:
            print(f"❌ Error in get_historical_prices: {str(e)}")
            return None
    
    def simulate_strategy_data_retrieval(self):
        """Simulate exactly what AdvancedLorentzianStrategy does to get data"""
        from lumibot.entities import Asset
        
        print(f"🎯 Simulating AdvancedLorentzianStrategy data retrieval...")
        
        # Create asset exactly like AdvancedLorentzianStrategy
        asset = Asset(self.symbol, Asset.AssetType.STOCK)
        
        # Use same parameters as AdvancedLorentzianStrategy
        max_bars_back = 2000  # Default from ClassifierSettings
        history_window = 2000  # Default from parameters  
        history_length = max(max_bars_back, history_window)
        
        print(f"   Calling get_historical_prices(asset='{self.symbol}', length={history_length}, timeframe='day')")
        print(f"   This is EXACTLY what AdvancedLorentzianStrategy does in on_trading_iteration()")
        
        # Make the exact same call as AdvancedLorentzianStrategy
        bars = self.get_historical_prices(asset, history_length, "day")
        
        if bars is not None:
            df = bars.df.copy()
            self.retrieved_data = df
            
            # Analyze what was retrieved
            print(f"✅ Strategy retrieved: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
            
            # Check relationship to backtest window
            backtest_start = datetime.strptime(os.getenv('BACKTESTING_START'), '%Y-%m-%d').date()
            backtest_end = datetime.strptime(os.getenv('BACKTESTING_END'), '%Y-%m-%d').date()
            
            data_start = df.index[0].date()
            data_end = df.index[-1].date()
            
            print(f"   Backtest window: {backtest_start} to {backtest_end}")
            print(f"   Data retrieved:  {data_start} to {data_end}")
            
            # Analyze data periods
            before_window = df[df.index.date < backtest_start]
            within_window = df[(df.index.date >= backtest_start) & (df.index.date <= backtest_end)]
            after_window = df[df.index.date > backtest_end]
            
            print(f"\n📊 Data Breakdown:")
            print(f"   BEFORE backtest window: {len(before_window)} rows")
            if len(before_window) > 0:
                print(f"      Range: {before_window.index[0].date()} to {before_window.index[-1].date()}")
                
            print(f"   WITHIN backtest window:  {len(within_window)} rows")
            if len(within_window) > 0:
                print(f"      Range: {within_window.index[0].date()} to {within_window.index[-1].date()}")
                
            print(f"   AFTER backtest window:   {len(after_window)} rows")
            if len(after_window) > 0:
                print(f"      Range: {after_window.index[0].date()} to {after_window.index[-1].date()}")
            
            # Key insights
            total_years = (data_end - data_start).days / 365.25
            within_pct = len(within_window) / len(df) * 100
            
            print(f"\n🔍 Key Insights:")
            print(f"   Data spans {total_years:.1f} years")
            print(f"   Only {within_pct:.1f}% of data is within backtest window")
            
            if len(before_window) > 0:
                print(f"   Strategy has access to {len(before_window)} days of PRE-backtest data")
                print(f"   This could create look-ahead bias or data mismatch with optimization")
                
            if len(after_window) > 0:
                print(f"   ⚠️  Strategy has {len(after_window)} days of FUTURE data!")
                print(f"       This is SERIOUS look-ahead bias!")
            
            return df
        else:
            print("❌ Strategy failed to retrieve data")
            return None

def run_lumibot_backtest_simulation(symbol):
    """Run a minimal backtest to see how Lumibot actually behaves"""
    from lumibot.strategies.strategy import Strategy
    from lumibot.backtesting import PolygonDataBacktesting
    from lumibot.entities import Asset
    
    print(f"🚀 Running minimal Lumibot backtest to test data behavior...")
    
    class DataExtractionStrategy(Strategy):
        """Minimal strategy that just extracts data like AdvancedLorentzianStrategy"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.extracted_data = None
            self.data_extraction_complete = False
        
        def initialize(self):
            self.sleeptime = "1D"  # Same as AdvancedLorentzianStrategy
        
        def on_trading_iteration(self):
            if not self.data_extraction_complete:
                try:
                    asset = Asset(symbol, Asset.AssetType.STOCK)
                    
                    # EXACT same call as AdvancedLorentzianStrategy
                    history_length = max(2000, 2000)  # max(maxBarsBack, history_window)
                    bars = self.get_historical_prices(asset, history_length, "day")
                    
                    if bars and len(bars.df) > 0:
                        self.extracted_data = bars.df.copy()
                        print(f"📊 Backtest extracted: {len(bars.df)} rows")
                        print(f"   Range: {bars.df.index[0].date()} to {bars.df.index[-1].date()}")
                        
                        # Compare with expected backtest window
                        backtest_start = datetime.strptime(os.getenv('BACKTESTING_START'), '%Y-%m-%d').date()
                        backtest_end = datetime.strptime(os.getenv('BACKTESTING_END'), '%Y-%m-%d').date()
                        
                        print(f"   Backtest window: {backtest_start} to {backtest_end}")
                        
                        # Check if this matches our direct test
                        data_start = bars.df.index[0].date()
                        data_end = bars.df.index[-1].date()
                        
                        if data_start < backtest_start:
                            days_before = (backtest_start - data_start).days
                            print(f"   📈 Has {days_before} days BEFORE backtest window")
                            
                        if data_end > backtest_end:
                            days_after = (data_end - backtest_end).days  
                            print(f"   ⚠️  Has {days_after} days AFTER backtest window (look-ahead bias!)")
                        
                        self.data_extraction_complete = True
                    else:
                        print("❌ No data extracted in backtest")
                        
                except Exception as e:
                    print(f"❌ Error during backtest data extraction: {str(e)}")
                    self.data_extraction_complete = True
    
    try:
        # Run the backtest
        start_dt = datetime.strptime(os.getenv('BACKTESTING_START'), '%Y-%m-%d')
        end_dt = datetime.strptime(os.getenv('BACKTESTING_END'), '%Y-%m-%d')
        
        strategy = DataExtractionStrategy()
        
        result = strategy.backtest(
            datasource_class=PolygonDataBacktesting,
            benchmark_asset=Asset(symbol, Asset.AssetType.STOCK),
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False
        )
        
        return strategy.extracted_data
        
    except Exception as e:
        print(f"❌ Backtest simulation failed: {str(e)}")
        return None

def compare_dataframes(df1, df2, source1_name, source2_name):
    """Compare two dataframes and identify differences"""
    print(f"\n🔍 DETAILED DATA COMPARISON")
    print("=" * 60)
    
    if df1 is None or df2 is None:
        print("❌ Cannot compare - one or both datasets are None")
        return
    
    # Basic info
    print(f"\n📊 Dataset Info:")
    print(f"   {source1_name}: {len(df1)} rows")
    print(f"   {source2_name}: {len(df2)} rows")
    
    # Date range comparison
    print(f"\n📅 Date Ranges:")
    print(f"   {source1_name}: {df1.index[0].date()} to {df1.index[-1].date()}")
    print(f"   {source2_name}: {df2.index[0].date()} to {df2.index[-1].date()}")
    
    # Check for the ROOT CAUSE we identified
    backtest_start = datetime.strptime(os.getenv('BACKTESTING_START'), '%Y-%m-%d').date()
    backtest_end = datetime.strptime(os.getenv('BACKTESTING_END'), '%Y-%m-%d').date()
    
    print(f"\n🎯 BACKTEST WINDOW ANALYSIS:")
    print(f"   Expected window: {backtest_start} to {backtest_end}")
    
    # Analyze each dataset's relationship to backtest window
    for df, name in [(df1, source1_name), (df2, source2_name)]:
        data_start = df.index[0].date()
        data_end = df.index[-1].date()
        
        within_window = df[(df.index.date >= backtest_start) & (df.index.date <= backtest_end)]
        before_window = df[df.index.date < backtest_start]
        after_window = df[df.index.date > backtest_end]
        
        print(f"\n   {name}:")
        print(f"     Total data: {data_start} to {data_end} ({len(df)} rows)")
        print(f"     Within window: {len(within_window)} rows ({len(within_window)/len(df)*100:.1f}%)")
        print(f"     Before window: {len(before_window)} rows")
        print(f"     After window: {len(after_window)} rows")
        
        if len(before_window) > 0:
            print(f"     📊 PRE-BACKTEST data available: {(backtest_start - data_start).days} days")
        if len(after_window) > 0:
            print(f"     ⚠️  POST-BACKTEST data available: {(data_end - backtest_end).days} days (LOOK-AHEAD BIAS!)")
    
    # Find overlapping dates
    common_dates = df1.index.intersection(df2.index)
    print(f"\n🔗 Common Dates: {len(common_dates)} dates")
    
    if len(common_dates) == 0:
        print("❌ No common dates found!")
        print("\n🚨 ROOT CAUSE CONFIRMED: The two data sources use completely different time periods!")
        print("   This explains why optimization performance != strategy performance")
        return
    
    # Compare overlapping data
    if len(common_dates) > 0:
        print(f"\n📈 Value Comparison (first 5 common dates):")
        for i, date in enumerate(common_dates[:5]):
            close1 = df1.loc[date, 'close']
            close2 = df2.loc[date, 'close']
            diff = abs(close1 - close2)
            print(f"   {date.date()}: {source1_name}=${close1:.2f}, {source2_name}=${close2:.2f}, diff=${diff:.4f}")
        
        # Statistical comparison
        df1_common = df1.loc[common_dates]
        df2_common = df2.loc[common_dates]
        
        close_corr = df1_common['close'].corr(df2_common['close'])
        avg_close_diff = abs(df1_common['close'] - df2_common['close']).mean()
        max_close_diff = abs(df1_common['close'] - df2_common['close']).max()
        
        print(f"\n📊 Statistical Analysis (overlapping data):")
        print(f"   Close price correlation: {close_corr:.6f}")
        print(f"   Average close difference: ${avg_close_diff:.4f}")
        print(f"   Maximum close difference: ${max_close_diff:.4f}")
        
        if close_corr > 0.999 and avg_close_diff < 0.01:
            print(f"   ✅ Price data is essentially identical for overlapping dates")
        else:
            print(f"   ⚠️  Price data differs between sources")

def clear_lumibot_cache():
    """Clear Lumibot cache to force fresh data download"""
    cache_path = r"C:\Users\khang\AppData\Local\LumiWealth\lumibot\Cache"
    
    print(f"🗑️  Clearing Lumibot cache to force fresh data download...")
    
    try:
        if os.path.exists(cache_path):
            # Remove cache directory
            shutil.rmtree(cache_path)
            print(f"✅ Cache cleared: {cache_path}")
            return True
        else:
            print(f"ℹ️  Cache directory not found: {cache_path}")
            return False
    except Exception as e:
        print(f"❌ Error clearing cache: {str(e)}")
        return False

def test_lumibot_with_fresh_cache(symbol, start_date, end_date):
    """Test Lumibot behavior after clearing cache"""
    print(f"\n🔬 Testing Lumibot with fresh cache (no old data)...")
    
    # Clear cache first
    cache_cleared = clear_lumibot_cache()
    
    if cache_cleared:
        print(f"   Cache cleared - Lumibot will download fresh data")
    else:
        print(f"   Cache not cleared - using existing data")
    
    # Now test Lumibot data retrieval
    test_strategy = TestStrategy(symbol)
    fresh_data = test_strategy.simulate_strategy_data_retrieval()
    
    if fresh_data is not None:
        print(f"✅ Fresh Lumibot data: {len(fresh_data)} rows from {fresh_data.index[0].date()} to {fresh_data.index[-1].date()}")
        
        # Check if fresh data matches backtest window better
        backtest_start = datetime.strptime(start_date, '%Y-%m-%d').date()
        backtest_end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        within_window = fresh_data[(fresh_data.index.date >= backtest_start) & (fresh_data.index.date <= backtest_end)]
        within_pct = len(within_window) / len(fresh_data) * 100
        
        print(f"   Data within backtest window: {len(within_window)} rows ({within_pct:.1f}%)")
        
        if within_pct > 50:
            print(f"   ✅ Fresh cache provides better backtest window coverage!")
        else:
            print(f"   ⚠️  Fresh cache still doesn't match backtest window well")
            
        return fresh_data
    else:
        print(f"❌ Failed to get fresh data from Lumibot")
        return None

def main():
    """Main comparison function"""
    print("🔍 COMPREHENSIVE DATA SOURCE INVESTIGATION")
    print("Lumibot Cache Behavior vs Direct Polygon API")
    print("=" * 60)
    
    # Get parameters from environment
    symbol = os.getenv('SYMBOL', 'TSLA')
    start_date = os.getenv('BACKTESTING_START', '2024-01-31')
    end_date = os.getenv('BACKTESTING_END', '2024-12-31')
    
    print(f"📋 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Backtest Window: {start_date} to {end_date}")
    print(f"   Goal: Compare optimization data vs strategy data")
    print()
    
    # Method 1: Direct Polygon API (used by optimization)
    print(f"🔬 METHOD 1: Direct Polygon API (Optimization Method)")
    print("-" * 50)
    polygon_data = download_polygon_direct(symbol, start_date, end_date)
    
    print(f"\n🔬 METHOD 2: Lumibot Strategy Simulation (Strategy Method)")
    print("-" * 50)
    
    # Method 2: Simulate AdvancedLorentzianStrategy data retrieval
    test_strategy = TestStrategy(symbol)
    lumibot_data = test_strategy.simulate_strategy_data_retrieval()
    
    # Method 3: Try actual backtest (if simulation works)
    if lumibot_data is not None:
        print(f"\n🔬 METHOD 3: Actual Lumibot Backtest (Verification)")
        print("-" * 50)
        backtest_data = run_lumibot_backtest_simulation(symbol)
        
        if backtest_data is not None:
            print(f"✅ Backtest verification: {len(backtest_data)} rows")
            
            # Quick compare with simulation
            if len(backtest_data) == len(lumibot_data):
                print(f"✅ Simulation matches backtest exactly")
            else:
                print(f"⚠️  Simulation differs from backtest: {len(lumibot_data)} vs {len(backtest_data)} rows")
    
    # Method 4: Test with fresh cache (clear old data)
    print(f"\n🔬 METHOD 4: Lumibot with Fresh Cache (Fix Attempt)")
    print("-" * 50)
    
    # Ask user if they want to clear cache (destructive operation)
    print(f"⚠️  WARNING: This will delete all Lumibot cached data!")
    print(f"   Your future backtests may be slower as data needs to be re-downloaded.")
    print(f"   But this should force Lumibot to use data within your backtest window.")
    
    response = input(f"\n   Clear Lumibot cache to test fresh data? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        fresh_lumibot_data = test_lumibot_with_fresh_cache(symbol, start_date, end_date)
        
        if fresh_lumibot_data is not None:
            # Compare fresh vs old Lumibot data
            print(f"\n📊 FRESH vs OLD Lumibot Data:")
            print(f"   Old cache: {len(lumibot_data)} rows ({lumibot_data.index[0].date()} to {lumibot_data.index[-1].date()})")
            print(f"   Fresh cache: {len(fresh_lumibot_data)} rows ({fresh_lumibot_data.index[0].date()} to {fresh_lumibot_data.index[-1].date()})")
            
            # Use fresh data for final comparison
            lumibot_data = fresh_lumibot_data
            print(f"   🔄 Using fresh data for final comparison")
    else:
        print(f"   Skipping cache clear - using existing cached data")
        print(f"   Note: This means Lumibot will continue using old cached data from 2023")
    
    # Compare the datasets
    print(f"\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    compare_dataframes(polygon_data, lumibot_data, "Polygon Direct (Optimization)", "Lumibot Strategy (Real)")
    
    # Final insights
    print(f"\n💡 KEY INSIGHTS:")
    print("=" * 30)
    
    if polygon_data is not None and lumibot_data is not None:
        polygon_days = len(polygon_data)
        lumibot_days = len(lumibot_data)
        
        print(f"1. SCALE DIFFERENCE:")
        print(f"   • Optimization uses {polygon_days} days of data")
        print(f"   • Strategy uses {lumibot_days} days of data ({lumibot_days/polygon_days:.1f}x more)")
        
        print(f"\n2. TIME PERIOD DIFFERENCE:")
        print(f"   • Optimization: {polygon_data.index[0].date()} to {polygon_data.index[-1].date()}")
        print(f"   • Strategy: {lumibot_data.index[0].date()} to {lumibot_data.index[-1].date()}")
        
        # Check for look-ahead bias
        backtest_end = datetime.strptime(end_date, '%Y-%m-%d').date()
        strategy_end = lumibot_data.index[-1].date()
        
        if strategy_end > backtest_end:
            days_future = (strategy_end - backtest_end).days
            print(f"\n3. LOOK-AHEAD BIAS:")
            print(f"   • Strategy has access to {days_future} days of FUTURE data!")
            print(f"   • This violates backtesting assumptions")
        
        print(f"\n4. SOLUTION:")
        print(f"   • Modify AdvancedLorentzianStrategy to limit data to backtest window")
        print(f"   • Or modify optimization to use same cache-based approach")
        print(f"   • Current mismatch explains performance differences")
    
    print(f"\n✅ Investigation complete!")

if __name__ == "__main__":
    main() 