"""Fetch historical news data from free APIs."""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Robust imports
import sys
import os
from pathlib import Path

# Get keys from environment (will be set from Colab secrets or keys.env)
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

try:
    from ..utils.config import RAW_DATA_DIR
    from ..utils.helpers import save_artifact, load_artifact
except ImportError:
    # Fallback for direct execution
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from utils.config import RAW_DATA_DIR
    from utils.helpers import save_artifact, load_artifact


class NewsFetcher:
    """Fetch and cache historical news data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.finnhub_key = FINNHUB_API_KEY
        self.newsapi_key = NEWS_API_KEY
    
    def fetch_finnhub(self, ticker: str, start_date: str, end_date: str,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch news from Finnhub (free tier: 60 calls/minute).
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with columns: date, headline, summary, source, url
        """
        if not self.finnhub_key:
            print("Warning: FINNHUB_API_KEY not set, skipping Finnhub")
            return pd.DataFrame()
        
        cache_file = self.cache_dir / f"finnhub_{ticker}_{start_date}_{end_date}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached Finnhub news for {ticker}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        print(f"Fetching Finnhub news for {ticker}...")
        
        try:
            import finnhub
            finnhub_client = finnhub.Client(api_key=self.finnhub_key)
            
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts = int(pd.Timestamp(end_date).timestamp())
            
            # Fetch company news
            news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
            
            if not news:
                print(f"No news found for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for item in news:
                records.append({
                    'date': pd.Timestamp(item.get('datetime', 0), unit='s'),
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'ticker': ticker
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df.to_csv(cache_file)
                print(f"✓ Fetched {len(df)} news articles for {ticker}")
            
            # Rate limiting (60 calls/min = 1 call/sec)
            time.sleep(1.1)
            
            return df
            
        except Exception as e:
            print(f"Error fetching Finnhub news for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_newsapi(self, query: str, start_date: str, end_date: str,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch news from NewsAPI (free tier: 100 requests/day).
        
        Args:
            query: Search query (e.g., "AAPL" or "Apple stock")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with news articles
        """
        if not self.newsapi_key:
            print("Warning: NEWS_API_KEY not set, skipping NewsAPI")
            return pd.DataFrame()
        
        cache_file = self.cache_dir / f"newsapi_{query}_{start_date}_{end_date}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached NewsAPI news for {query}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        print(f"Fetching NewsAPI news for {query}...")
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': start_date,
                'to': end_date,
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key,
                'language': 'en',
                'pageSize': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
            
            articles = data.get('articles', [])
            if not articles:
                print(f"No news found for {query}")
                return pd.DataFrame()
            
            records = []
            for article in articles:
                records.append({
                    'date': pd.to_datetime(article.get('publishedAt', '')),
                    'headline': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'query': query
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values('date').reset_index(drop=True)
                df.to_csv(cache_file)
                print(f"✓ Fetched {len(df)} news articles for {query}")
            
            # Rate limiting (100/day = be conservative)
            time.sleep(2)
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"⚠ NewsAPI rate limit reached (429) - skipping NewsAPI")
            elif e.response.status_code == 401:
                print(f"⚠ NewsAPI authentication failed - check API key")
            else:
                print(f"⚠ NewsAPI HTTP error: {e}")
            return pd.DataFrame()
        except Exception as e:
            error_msg = str(e).lower()
            if "limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                print(f"⚠ NewsAPI limit/quota reached - skipping NewsAPI")
            else:
                print(f"⚠ Error fetching NewsAPI news for {query}: {e}")
            return pd.DataFrame()
    
    def fetch_all(self, ticker: str, start_date: str, end_date: str,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch news from all available sources and combine.
        Prioritizes Finnhub (more reliable) and gracefully handles NewsAPI limits.
        """
        dfs = []
        
        # PRIORITY 1: Try Finnhub first (more reliable, better free tier)
        print("\n[Priority 1] Fetching from Finnhub...")
        df_finnhub = self.fetch_finnhub(ticker, start_date, end_date, use_cache)
        if not df_finnhub.empty:
            dfs.append(df_finnhub)
            print(f"✓ Got {len(df_finnhub)} articles from Finnhub")
        
        # PRIORITY 2: Try NewsAPI only if we have limited data (gracefully handle limits)
        if len(dfs) == 0 or len(df_finnhub) < 50:  # Only use NewsAPI if we need more data
            print("\n[Priority 2] Fetching from NewsAPI (if available)...")
            try:
                df_newsapi = self.fetch_newsapi(ticker, start_date, end_date, use_cache)
                if not df_newsapi.empty:
                    dfs.append(df_newsapi)
                    print(f"✓ Got {len(df_newsapi)} articles from NewsAPI")
            except Exception as e:
                # Gracefully handle NewsAPI limits/errors
                if "limit" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                    print(f"⚠ NewsAPI limit reached - continuing with Finnhub data only")
                else:
                    print(f"⚠ NewsAPI error (non-critical): {e}")
        
        if not dfs:
            print("⚠ No news data fetched from any source")
            return pd.DataFrame()
        
        # Combine and deduplicate
        print("\nCombining and deduplicating news from all sources...")
        combined = pd.concat(dfs, ignore_index=True)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['headline', 'date'], keep='first')
        after_dedup = len(combined)
        combined = combined.sort_values('date').reset_index(drop=True)
        
        if before_dedup > after_dedup:
            print(f"✓ Removed {before_dedup - after_dedup} duplicate articles")
        
        print(f"✓ Total unique articles: {len(combined)}")
        
        return combined


if __name__ == "__main__":
    # Test
    fetcher = NewsFetcher()
    df = fetcher.fetch_all('AAPL', '2023-01-01', '2023-12-31')
    print(f"\nFetched {len(df)} news articles")
    if not df.empty:
        print(df.head())

