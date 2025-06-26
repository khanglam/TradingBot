"""
Lumibot Cache Utilities
=======================

This module provides utilities to access data cached by Lumibot's PolygonDataBacktesting.
It allows test_parameters.py and optimize_parameters.py to use the same cached data
that AdvancedLorentzianStrategy uses, ensuring consistency and avoiding repeated API calls.

The cache location is typically: C:\Users\{username}\AppData\Local\LumiWealth\lumibot\Cache\1.0\polygon
"""

import os
import pandas as pd
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class LumibotCacheUtils:
    """Utilities for accessing Lumibot's cached data"""
    
    def __init__(self):
        self.cache_base_path = self._get_cache_path()
        self.timeframe_mapping = {
            # Map our DATA_TIMEFRAME to Lumibot's format
            'minute': '1M',  # 1 minute
            '1M': '1M',
            'hour': '1H',    # 1 hour  
            '1H': '1H',
            'day': '1D',     # 1 day (default)
            '1D': '1D',
            'daily': '1D',
        }
        
        # Map DATA_TIMEFRAME to sleeptime equivalent
        self.sleeptime_mapping = {
            'minute': '1M',
            '1M': '1M', 
            'hour': '1H',
            '1H': '1H',
            'day': '1D',
            '1D': '1D',
            'daily': '1D',
        }
    
    def _get_cache_path(self) -> Optional[Path]:
        """Get the Lumibot cache directory path"""
        # Common cache locations
        possible_paths = [
            # Windows
            Path.home() / "AppData" / "Local" / "LumiWealth" / "lumibot" / "Cache" / "1.0" / "polygon",
            # Alternative Windows location
            Path.home() / ".lumibot" / "cache" / "polygon",
            # Linux/Mac
            Path.home() / ".cache" / "lumibot" / "polygon",
            Path.home() / ".lumibot" / "cache" / "polygon",
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"✅ Found Lumibot cache at: {path}")
                return path
        
        # If no cache found, try to create directory structure
        default_path = Path.home() / "AppData" / "Local" / "LumiWealth" / "lumibot" / "Cache" / "1.0" / "polygon"
        print(f"❌ No existing cache found, will check: {default_path}")
        return default_path
    
    def map_timeframe(self, timeframe: str) -> str:
        """Map DATA_TIMEFRAME to Lumibot format"""
        return self.timeframe_mapping.get(timeframe.lower(), '1D')
    
    def map_to_sleeptime(self, timeframe: str) -> str:
        """Map DATA_TIMEFRAME to sleeptime format"""
        return self.sleeptime_mapping.get(timeframe.lower(), '1D')
    
    def _generate_cache_key(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> str:
        """Generate cache key similar to how Lumibot does it"""
        # Convert timeframe to Lumibot format
        lumibot_timeframe = self.map_timeframe(timeframe)
        
        # Create a hash-based cache key (similar to Lumibot's approach)
        cache_string = f"{symbol}_{start_date}_{end_date}_{lumibot_timeframe}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _search_cache_files(self, symbol: str, timeframe: str) -> list:
        """Search for cache files that match the symbol and timeframe"""
        if not self.cache_base_path or not self.cache_base_path.exists():
            return []
        
        lumibot_timeframe = self.map_timeframe(timeframe)
        matching_files = []
        
        # Search all files in cache directory
        for file_path in self.cache_base_path.rglob("*.pkl"):
            # Check if filename contains our symbol and timeframe indicators
            filename = file_path.name.lower()
            if symbol.lower() in filename and any(tf in filename for tf in [lumibot_timeframe.lower(), 'day', 'daily']):
                matching_files.append(file_path)
        
        # Sort by modification time (newest first)
        matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matching_files
    
    def load_cached_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = 'day') -> Optional[pd.DataFrame]:
        """
        Load data from Lumibot's cache
        
        Args:
            symbol: Stock symbol (e.g., 'TSLA')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe ('day', 'hour', 'minute')
        
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        print(f"🔍 Searching Lumibot cache for {symbol} ({timeframe}) from {start_date} to {end_date}...")
        
        if not self.cache_base_path:
            print(f"❌ Cache path not found")
            return None
        
        # Search for matching cache files
        matching_files = self._search_cache_files(symbol, timeframe)
        
        if not matching_files:
            print(f"❌ No cached data found for {symbol} with timeframe {timeframe}")
            print(f"   Cache directory: {self.cache_base_path}")
            if self.cache_base_path.exists():
                all_files = list(self.cache_base_path.rglob("*.pkl"))
                print(f"   Total cache files: {len(all_files)}")
                if all_files:
                    print(f"   Sample files: {[f.name for f in all_files[:3]]}")
            return None
        
        # Try to load the most recent matching file
        for cache_file in matching_files:
            try:
                print(f"📂 Trying cache file: {cache_file.name}")
                
                # Load the pickle file
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Extract DataFrame from cached data
                df = self._extract_dataframe_from_cache(cached_data)
                
                if df is not None and not df.empty:
                    # Filter data by date range
                    df_filtered = self._filter_by_date_range(df, start_date, end_date)
                    
                    if df_filtered is not None and len(df_filtered) > 0:
                        print(f"✅ Loaded {len(df_filtered)} rows from Lumibot cache")
                        print(f"   Date range: {df_filtered.index[0]} to {df_filtered.index[-1]}")
                        print(f"   Cache file: {cache_file.name}")
                        return df_filtered
                    else:
                        print(f"⚠️  Cache file doesn't contain data for requested date range")
                        continue
                else:
                    print(f"⚠️  Cache file doesn't contain valid DataFrame")
                    continue
                    
            except Exception as e:
                print(f"⚠️  Failed to load cache file {cache_file.name}: {e}")
                continue
        
        print(f"❌ No usable cached data found for {symbol}")
        return None
    
    def _extract_dataframe_from_cache(self, cached_data: Any) -> Optional[pd.DataFrame]:
        """Extract DataFrame from Lumibot's cached data structure"""
        try:
            # Lumibot cache can have different structures, try to handle them
            if isinstance(cached_data, pd.DataFrame):
                return cached_data
            
            elif isinstance(cached_data, dict):
                # Look for DataFrame in dict values
                for key, value in cached_data.items():
                    if isinstance(value, pd.DataFrame) and not value.empty:
                        # Check if it looks like OHLCV data
                        if all(col in value.columns for col in ['open', 'high', 'low', 'close']):
                            return value
                        elif all(col.lower() in [c.lower() for c in value.columns] for col in ['open', 'high', 'low', 'close']):
                            # Convert to lowercase columns
                            value.columns = [col.lower() for col in value.columns]
                            return value
            
            elif isinstance(cached_data, list) and cached_data:
                # Try to convert list of bars to DataFrame
                if hasattr(cached_data[0], 'open'):  # Lumibot Bar objects
                    data_list = []
                    for bar in cached_data:
                        data_list.append({
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': getattr(bar, 'volume', 0),
                            'date': bar.time if hasattr(bar, 'time') else bar.timestamp
                        })
                    
                    df = pd.DataFrame(data_list)
                    df.set_index('date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    return df
                
            print(f"⚠️  Unknown cache data structure: {type(cached_data)}")
            return None
            
        except Exception as e:
            print(f"❌ Error extracting DataFrame from cache: {e}")
            return None
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Filter DataFrame by date range"""
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Parse date strings
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter by date range
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            filtered_df = df.loc[mask].copy()
            
            if filtered_df.empty:
                print(f"⚠️  No data found in date range {start_date} to {end_date}")
                print(f"   Available data range: {df.index.min()} to {df.index.max()}")
                return None
            
            return filtered_df
            
        except Exception as e:
            print(f"❌ Error filtering by date range: {e}")
            return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        info = {
            'cache_path': str(self.cache_base_path) if self.cache_base_path else None,
            'cache_exists': self.cache_base_path.exists() if self.cache_base_path else False,
            'total_files': 0,
            'total_size_mb': 0,
            'symbols_found': set(),
            'timeframes_found': set()
        }
        
        if self.cache_base_path and self.cache_base_path.exists():
            cache_files = list(self.cache_base_path.rglob("*.pkl"))
            info['total_files'] = len(cache_files)
            
            total_size = sum(f.stat().st_size for f in cache_files)
            info['total_size_mb'] = total_size / (1024 * 1024)
            
            # Extract symbols and timeframes from filenames
            for file_path in cache_files:
                filename = file_path.name.lower()
                # Try to extract symbol patterns
                for symbol in ['tsla', 'aapl', 'msft', 'googl', 'spy', 'qqq']:
                    if symbol in filename:
                        info['symbols_found'].add(symbol.upper())
                
                # Look for timeframe patterns
                for tf in ['1d', 'day', '1h', 'hour', '1m', 'minute']:
                    if tf in filename:
                        info['timeframes_found'].add(tf)
        
        return info

# Global cache utils instance
cache_utils = LumibotCacheUtils()

def load_lumibot_cached_data(symbol: str, start_date: str, end_date: str, timeframe: str = 'day') -> Optional[pd.DataFrame]:
    """
    Convenience function to load data from Lumibot's cache
    
    This function provides the same interface as download_real_data() but uses
    Lumibot's cache instead of making API calls.
    
    Args:
        symbol: Stock symbol (e.g., 'TSLA')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timeframe: Data timeframe ('day', 'hour', 'minute')
    
    Returns:
        DataFrame with OHLCV data or None if not found
    """
    return cache_utils.load_cached_data(symbol, start_date, end_date, timeframe)

def map_timeframe_to_sleeptime(timeframe: str) -> str:
    """
    Map DATA_TIMEFRAME environment variable to Lumibot sleeptime format
    
    This ensures DATA_TIMEFRAME is equivalent to self.sleeptime in AdvancedLorentzianStrategy
    
    Args:
        timeframe: timeframe string ('day', 'hour', 'minute', etc.)
        
    Returns:
        Lumibot sleeptime format ('1D', '1H', '1M')
    """
    return cache_utils.map_to_sleeptime(timeframe)

def display_cache_info():
    """Display information about Lumibot's cache"""
    info = cache_utils.get_cache_info()
    
    print("\n" + "="*60)
    print("📂 LUMIBOT CACHE INFORMATION")
    print("="*60)
    
    print(f"📁 Cache Path: {info['cache_path']}")
    print(f"✅ Cache Exists: {info['cache_exists']}")
    
    if info['cache_exists']:
        print(f"📊 Total Files: {info['total_files']}")
        print(f"💾 Total Size: {info['total_size_mb']:.1f} MB")
        
        if info['symbols_found']:
            print(f"📈 Symbols Found: {', '.join(sorted(info['symbols_found']))}")
        
        if info['timeframes_found']:
            print(f"⏰ Timeframes Found: {', '.join(sorted(info['timeframes_found']))}")
        
        if info['total_files'] == 0:
            print("⚠️  No cache files found - run AdvancedLorentzianStrategy backtest first")
    else:
        print("❌ Cache directory not found")
        print("💡 Run AdvancedLorentzianStrategy backtest first to populate cache")
    
    print("="*60)

if __name__ == "__main__":
    # Test the cache utils
    display_cache_info()
    
    # Test data loading
    test_symbol = 'TSLA'
    test_start = '2024-01-01'
    test_end = '2024-12-31'
    
    print(f"\n🧪 Testing cache data loading for {test_symbol}...")
    df = load_lumibot_cached_data(test_symbol, test_start, test_end, 'day')
    
    if df is not None:
        print(f"✅ Successfully loaded {len(df)} rows of data")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    else:
        print(f"❌ No data found in cache for {test_symbol}") 