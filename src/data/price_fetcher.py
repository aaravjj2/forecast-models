"""Fetch historical stock price data from free APIs."""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
# Handle both relative and absolute imports
try:
    from ..utils.config import RAW_DATA_DIR
    from ..utils.helpers import save_artifact, load_artifact
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import RAW_DATA_DIR
    from utils.helpers import save_artifact, load_artifact


class PriceFetcher:
    """Fetch and cache historical OHLCV data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "prices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(self, ticker: str, start_date: str, end_date: str, 
              use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"
        
        # Check cache
        if use_cache and cache_file.exists():
            print(f"Loading cached data for {ticker}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        # Fetch from yfinance (free, no API key needed)
        print(f"Fetching price data for {ticker} from {start_date} to {end_date}...")
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Standardize column names
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['Date'])
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df.set_index('date').sort_index()
            
            # Remove any duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Save to cache
            df.to_csv(cache_file)
            print(f"✓ Fetched {len(df)} days of data for {ticker}")
            
            # Rate limiting
            time.sleep(0.5)
            
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            raise
    
    def fetch_multiple(self, tickers: List[str], start_date: str, end_date: str,
                      use_cache: bool = True) -> dict:
        """Fetch data for multiple tickers."""
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = self.fetch(ticker, start_date, end_date, use_cache)
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
                continue
        return data
    
    def fetch_index(self, index_symbol: str = '^GSPC', start_date: str = None, 
                   end_date: str = None) -> pd.DataFrame:
        """Fetch index data (e.g., S&P 500) for market context."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.fetch(index_symbol, start_date, end_date)


if __name__ == "__main__":
    # Test
    fetcher = PriceFetcher()
    df = fetcher.fetch('AAPL', '2020-01-01', '2023-12-31')
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

