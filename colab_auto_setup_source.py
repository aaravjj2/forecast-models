# AUTO-CREATE: Build all source files in Colab
# This creates the entire project structure without manual uploads

from pathlib import Path
import os

project_dir = Path('/content/ml_research_pipeline')
os.chdir(project_dir)

# Create all source files
files_content = {
    'src/utils/config.py': '''\"\"\"Configuration management for the pipeline.\"\"\"

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / "keys.env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Try loading from current directory or parent
    load_dotenv()

# API Keys - Load from environment (set from Colab secrets or keys.env)
# All keys match the names in keys.env
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB2_API_KEY = os.getenv("FINNHUB2_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
SEC_API_KEY = os.getenv("SEC_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
FINAGE_API_KEY = os.getenv("FINAGE_API_KEY", "")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SPECIALIST_MODELS_DIR = MODELS_DIR / "specialists"
META_MODEL_DIR = MODELS_DIR / "meta"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR, 
                 META_MODEL_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
DEFAULT_TRAIN_WINDOW_DAYS = 252  # 1 year
DEFAULT_TEST_WINDOW_DAYS = 21    # 1 month
DEFAULT_MIN_CONFIDENCE = 0.6     # Abstention threshold

# Feature engineering
LOOKBACK_WINDOWS = [1, 3, 5, 10, 20, 50]  # Days for returns/volatility
MA_WINDOWS = [5, 10, 20, 50, 200]  # Moving average windows

# Backtesting
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001  # 0.1%

''',
    'src/utils/helpers.py': '''\"\"\"Helper functions for data processing and validation.\"\"\"

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def validate_dataframe(df: pd.DataFrame, required_columns: list, 
                      check_nulls: bool = True) -> bool:
    \"\"\"Validate dataframe has required columns and optionally no nulls.\"\"\"
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if check_nulls:
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            print(f"Warning: Null values found:\\n{null_counts[null_counts > 0]}")
    
    return True


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fill_value: float = 0.0) -> pd.Series:
    \"\"\"Safely divide two series, handling zeros and nulls.\"\"\"
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill_value)


def align_time_series(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    \"\"\"Remove any future data leakage by filtering to dates <= target_date.\"\"\"
    if 'date' in df.columns:
        return df[df['date'] <= target_date].copy()
    elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
        return df[df.index <= target_date].copy()
    return df


def calculate_returns(prices: pd.Series, periods: list = [1, 3, 5, 10]) -> pd.DataFrame:
    \"\"\"Calculate returns over multiple periods.\"\"\"
    returns = pd.DataFrame(index=prices.index)
    for period in periods:
        returns[f'return_{period}d'] = prices.pct_change(period)
    return returns


def ensure_no_leakage(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                     date_col: str = 'date') -> tuple:
    \"\"\"Ensure no data leakage between train and test sets.\"\"\"
    if date_col in train_data.columns:
        max_train_date = train_data[date_col].max()
        min_test_date = test_data[date_col].min()
    else:
        max_train_date = train_data.index.max()
        min_test_date = test_data.index.min()
    
    if max_train_date >= min_test_date:
        raise ValueError(f"Data leakage detected! Train max: {max_train_date}, Test min: {min_test_date}")
    
    return train_data, test_data


def save_artifact(data: Any, filepath: str, format: str = 'pickle'):
    \"\"\"Save data artifact in specified format.\"\"\"
    from pathlib import Path
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=True)
        else:
            raise ValueError("CSV format only supports DataFrames")
    elif format == 'json':
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_artifact(filepath: str, format: str = 'pickle'):
    \"\"\"Load data artifact from specified format.\"\"\"
    from pathlib import Path
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    
    if format == 'pickle':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        return pd.read_csv(path, index_col=0, parse_dates=True)
    elif format == 'json':
        import json
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


''',
    'src/data/price_fetcher.py': '''\"\"\"Fetch historical stock price data from free APIs.\"\"\"

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
    \"\"\"Fetch and cache historical OHLCV data.\"\"\"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "prices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(self, ticker: str, start_date: str, end_date: str, 
              use_cache: bool = True) -> pd.DataFrame:
        \"\"\"
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        \"\"\"
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
        \"\"\"Fetch data for multiple tickers.\"\"\"
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
        \"\"\"Fetch index data (e.g., S&P 500) for market context.\"\"\"
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
    print(f"\\nShape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

''',
    'src/data/news_fetcher.py': '''\"\"\"Fetch historical news data from free APIs.\"\"\"

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
from pathlib import Path
try:
    from ..utils.config import RAW_DATA_DIR
import os
# Get keys from environment (will be set from Colab secrets or keys.env)
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    from ..utils.helpers import save_artifact, load_artifact
except ImportError:
    # Fallback for direct execution
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from utils.config import RAW_DATA_DIR, FINNHUB_API_KEY, NEWS_API_KEY
    from utils.helpers import save_artifact, load_artifact


class NewsFetcher:
    \"\"\"Fetch and cache historical news data.\"\"\"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.finnhub_key = FINNHUB_API_KEY
        self.newsapi_key = NEWS_API_KEY
    
    def fetch_finnhub(self, ticker: str, start_date: str, end_date: str,
                     use_cache: bool = True) -> pd.DataFrame:
        \"\"\"
        Fetch news from Finnhub (free tier: 60 calls/minute).
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with columns: date, headline, summary, source, url
        \"\"\"
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
        \"\"\"
        Fetch news from NewsAPI (free tier: 100 requests/day).
        
        Args:
            query: Search query (e.g., "AAPL" or "Apple stock")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with news articles
        \"\"\"
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
            
        except Exception as e:
            print(f"Error fetching NewsAPI news for {query}: {e}")
            return pd.DataFrame()
    
    def fetch_all(self, ticker: str, start_date: str, end_date: str,
                 use_cache: bool = True) -> pd.DataFrame:
        \"\"\"Fetch news from all available sources and combine.\"\"\"
        dfs = []
        
        # Try Finnhub
        df_finnhub = self.fetch_finnhub(ticker, start_date, end_date, use_cache)
        if not df_finnhub.empty:
            dfs.append(df_finnhub)
        
        # Try NewsAPI
        df_newsapi = self.fetch_newsapi(ticker, start_date, end_date, use_cache)
        if not df_newsapi.empty:
            dfs.append(df_newsapi)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine and deduplicate
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['headline', 'date'], keep='first')
        combined = combined.sort_values('date').reset_index(drop=True)
        
        return combined


if __name__ == "__main__":
    # Test
    fetcher = NewsFetcher()
    df = fetcher.fetch_all('AAPL', '2023-01-01', '2023-12-31')
    print(f"\\nFetched {len(df)} news articles")
    if not df.empty:
        print(df.head())

''',
    'src/features/feature_builder.py': '''\"\"\"Build features from price and news data.\"\"\"

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Robust imports
import sys
from pathlib import Path
try:
    from ..utils.config import LOOKBACK_WINDOWS, MA_WINDOWS
    from ..utils.helpers import safe_divide, calculate_returns, align_time_series
except ImportError:
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from utils.config import LOOKBACK_WINDOWS, MA_WINDOWS
    from utils.helpers import safe_divide, calculate_returns, align_time_series


class FeatureBuilder:
    \"\"\"Build time-aligned, leak-free features from price and news data.\"\"\"
    
    def __init__(self):
        self.feature_metadata = {}
    
    def build_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Build price-based features.
        
        Args:
            prices: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            DataFrame with engineered features
        \"\"\"
        df = prices.copy()
        
        # Returns over multiple periods
        for period in LOOKBACK_WINDOWS:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Volatility measures
        for period in [5, 10, 20]:
            df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std()
            df[f'realized_vol_{period}d'] = df['return_1d'].rolling(period).std() * np.sqrt(252)
        
        # Moving averages
        for window in MA_WINDOWS:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_{window}_ratio'] = safe_divide(df['close'], df[f'ma_{window}'])
            df[f'ma_{window}_slope'] = df[f'ma_{window}'].diff(5) / df[f'ma_{window}'].shift(5)
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_30'] = self._calculate_rsi(df['close'], 30)
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_30'] = self._calculate_atr(df, 30)
        
        # Price position in range
        for window in [20, 50]:
            df[f'high_{window}'] = df['high'].rolling(window).max()
            df[f'low_{window}'] = df['low'].rolling(window).min()
            df[f'price_position_{window}'] = safe_divide(
                df['close'] - df[f'low_{window}'],
                df[f'high_{window}'] - df[f'low_{window}']
            )
        
        # Volume features
        for window in [10, 20, 50]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = safe_divide(df['volume'], df[f'volume_ma_{window}'])
        
        # Price change features
        df['high_low_ratio'] = safe_divide(df['high'], df['low'])
        df['close_open_ratio'] = safe_divide(df['close'], df['open'])
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
        
        return df
    
    def build_market_features(self, prices: pd.DataFrame, 
                             index_prices: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Build market context features using index data.
        
        Args:
            prices: Stock price DataFrame
            index_prices: Index price DataFrame (e.g., S&P 500)
        
        Returns:
            DataFrame with market features
        \"\"\"
        df = prices.copy()
        
        # Align index data
        index_aligned = index_prices.reindex(prices.index, method='ffill')
        
        # Index returns
        for period in [1, 3, 5, 10]:
            df[f'index_return_{period}d'] = index_aligned['close'].pct_change(period)
        
        # Relative strength (stock vs index)
        for period in [5, 10, 20]:
            stock_return = df['close'].pct_change(period)
            index_return = index_aligned['close'].pct_change(period)
            df[f'relative_strength_{period}d'] = stock_return - index_return
        
        # Beta approximation (rolling correlation * volatility ratio)
        for window in [20, 60]:
            stock_returns = df['return_1d']
            index_returns = index_aligned['close'].pct_change()
            rolling_corr = stock_returns.rolling(window).corr(index_returns)
            vol_ratio = (stock_returns.rolling(window).std() / 
                        index_returns.rolling(window).std())
            df[f'beta_approx_{window}'] = rolling_corr * vol_ratio
        
        return df
    
    def build_news_features(self, prices: pd.DataFrame, 
                           news: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Build news-based features.
        
        Args:
            prices: Stock price DataFrame (indexed by date)
            news: News DataFrame with columns: date, headline, summary
        
        Returns:
            DataFrame with news features
        \"\"\"
        df = prices.copy()
        
        if news.empty:
            # Fill with zeros if no news
            df['news_count_1d'] = 0
            df['news_count_7d'] = 0
            df['news_sentiment_7d'] = 0
            return df
        
        # Align news to price dates
        news_aligned = news.set_index('date')
        news_aligned = news_aligned.reindex(df.index, fill_value=0)
        
        # News count features
        for window in [1, 3, 7, 14]:
            news_count = news_aligned.groupby(news_aligned.index).size()
            df[f'news_count_{window}d'] = news_count.rolling(window, min_periods=0).sum()
        
        # Simple sentiment (placeholder - will be enhanced with FinBERT)
        # For now, use keyword-based sentiment
        if 'headline' in news.columns:
            news['simple_sentiment'] = news['headline'].apply(self._simple_sentiment)
            news_sentiment = news.groupby('date')['simple_sentiment'].mean()
            news_sentiment_aligned = news_sentiment.reindex(df.index, method='ffill', fill_value=0)
            
            for window in [3, 7, 14]:
                df[f'news_sentiment_{window}d'] = news_sentiment_aligned.rolling(window, min_periods=0).mean()
        else:
            for window in [3, 7, 14]:
                df[f'news_sentiment_{window}d'] = 0
        
        return df
    
    def build_all_features(self, prices: pd.DataFrame, 
                          index_prices: Optional[pd.DataFrame] = None,
                          news: Optional[pd.DataFrame] = None,
                          target_col: str = 'close') -> pd.DataFrame:
        \"\"\"
        Build all features and create target variable.
        
        Args:
            prices: Price DataFrame
            index_prices: Optional index DataFrame
            news: Optional news DataFrame
            target_col: Column to use for target (default: 'close')
        
        Returns:
            DataFrame with all features and target
        \"\"\"
        # Start with price features
        df = self.build_price_features(prices)
        
        # Add market features if available
        if index_prices is not None and not index_prices.empty:
            df = self.build_market_features(df, index_prices)
        
        # Add news features if available
        if news is not None and not news.empty:
            df = self.build_news_features(df, news)
        
        # Create target: next day return (forward-looking, will be shifted in training)
        df['target_return_1d'] = df[target_col].pct_change(1).shift(-1)
        
        # Create binary target: 1 if return > 0, -1 if return < 0
        df['target_direction'] = np.where(df['target_return_1d'] > 0, 1, 
                                         np.where(df['target_return_1d'] < 0, -1, 0))
        
        # Fill NaN values with 0 (from rolling calculations at start)
        # Only drop rows where target is NaN (can't predict without target)
        df = df.fillna(0)
        df = df.dropna(subset=['target_return_1d', 'target_direction'])
        
        # Store feature metadata
        feature_cols = [col for col in df.columns 
                       if col not in ['target_return_1d', 'target_direction', 
                                     'open', 'high', 'low', 'close', 'volume']]
        self.feature_metadata = {
            'feature_columns': feature_cols,
            'target_columns': ['target_return_1d', 'target_direction'],
            'n_features': len(feature_cols)
        }
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        \"\"\"Calculate Relative Strength Index.\"\"\"
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = safe_divide(gain, loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        \"\"\"Calculate Average True Range.\"\"\"
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _simple_sentiment(self, text: str) -> float:
        \"\"\"Simple keyword-based sentiment (placeholder for FinBERT).\"\"\"
        if pd.isna(text):
            return 0.0
        
        text_lower = str(text).lower()
        
        # Positive keywords
        positive_words = ['up', 'gain', 'rise', 'surge', 'rally', 'bullish', 
                         'growth', 'profit', 'beat', 'strong', 'positive']
        # Negative keywords
        negative_words = ['down', 'fall', 'drop', 'decline', 'crash', 'bearish',
                         'loss', 'miss', 'weak', 'negative', 'warn']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count + 1)


if __name__ == "__main__":
    # Test
    import yfinance as yf
    prices = yf.download('AAPL', start='2023-01-01', end='2023-12-31', progress=False)
    prices = prices.reset_index()
    prices['date'] = pd.to_datetime(prices['Date'])
    prices = prices[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    prices.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    prices = prices.set_index('date')
    
    builder = FeatureBuilder()
    features = builder.build_all_features(prices)
    print(f"Built {len(features.columns)} features")
    print(f"Feature columns: {len(builder.feature_metadata['feature_columns'])}")
    print(features.head())

''',
    'src/models/base_model.py': '''\"\"\"Base class for all prediction models.\"\"\"

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class ModelSignal:
    \"\"\"Standardized model output.\"\"\"
    signal: int  # -1 (sell), 0 (abstain), 1 (buy)
    prob_up: float  # Probability of positive return [0, 1]
    confidence: float  # Model confidence [0, 1]
    raw_output: Optional[Any] = None  # Raw model output for debugging


class BaseModel(ABC):
    \"\"\"Base class that all specialist models must implement.\"\"\"
    
    def __init__(self, name: str, min_confidence: float = 0.6):
        self.name = name
        self.min_confidence = min_confidence
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        \"\"\"
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable (binary: -1, 0, 1)
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary with training metrics
        \"\"\"
        pass
    
    @abstractmethod
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        \"\"\"
        Get prediction signal for a single snapshot.
        
        Args:
            snapshot: Single row of features (pd.Series)
        
        Returns:
            ModelSignal with prediction
        \"\"\"
        pass
    
    @abstractmethod
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Predict on a batch of samples.
        
        Args:
            X: Feature matrix
        
        Returns:
            DataFrame with columns: signal, prob_up, confidence
        \"\"\"
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        \"\"\"Save model to disk.\"\"\"
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        \"\"\"Load model from disk.\"\"\"
        pass
    
    def _apply_abstention(self, prob_up: float, confidence: float) -> int:
        \"\"\"
        Apply abstention logic based on confidence threshold.
        
        Returns:
            signal: -1, 0, or 1 (0 = abstain)
        \"\"\"
        if confidence < self.min_confidence:
            return 0  # Abstain
        
        if prob_up > 0.5:
            return 1  # Buy
        elif prob_up < 0.5:
            return -1  # Sell
        else:
            return 0  # Abstain (uncertain)
    
    def _calculate_confidence(self, prob_up: float) -> float:
        \"\"\"
        Calculate confidence from probability.
        Higher confidence when probability is further from 0.5.
        \"\"\"
        return 2 * abs(prob_up - 0.5)  # Maps [0.5, 1.0] -> [0, 1]


''',
    'src/models/xgboost_model.py': '''\"\"\"XGBoost specialist model for price pattern learning.\"\"\"

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
from typing import Dict, Any

from .base_model import BaseModel, ModelSignal


class XGBoostModel(BaseModel):
    \"\"\"XGBoost model for price-based predictions.\"\"\"
    
    def __init__(self, min_confidence: float = 0.6, **xgb_params):
        super().__init__("XGBoost", min_confidence)
        
        # Default XGBoost parameters (free-tier friendly)
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',  # Faster, works on CPU
            'base_score': 0.5,  # Fix for logistic loss
            **xgb_params
        }
        
        self.model = None
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        \"\"\"Train XGBoost model.\"\"\"
        # Convert y to binary (0/1) for XGBoost
        y_binary = (y > 0).astype(int)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        train_acc = ((train_pred > 0.5) == y_train).mean()
        val_acc = ((val_pred > 0.5) == y_val).mean()
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'n_features': len(self.feature_columns),
            'n_estimators': self.model.num_boosted_rounds()
        }
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        \"\"\"Get prediction for a single snapshot.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X = pd.DataFrame([snapshot[self.feature_columns]])
        dtest = xgb.DMatrix(X)
        
        prob_up = float(self.model.predict(dtest)[0])
        confidence = self._calculate_confidence(prob_up)
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'prob_up': prob_up}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Predict on a batch.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X_aligned = X[self.feature_columns]
        dtest = xgb.DMatrix(X_aligned)
        
        prob_up = self.model.predict(dtest)
        confidence = np.array([self._calculate_confidence(p) for p in prob_up])
        signal = np.array([self._apply_abstention(p, c) 
                          for p, c in zip(prob_up, confidence)])
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        \"\"\"Save model and metadata.\"\"\"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'params': self.params,
            'min_confidence': self.min_confidence
        }
        joblib.dump(metadata, str(path).replace('.model', '_metadata.pkl'))
    
    def load(self, filepath: str):
        \"\"\"Load model and metadata.\"\"\"
        path = Path(filepath)
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        
        # Load metadata
        metadata = joblib.load(str(path).replace('.model', '_metadata.pkl'))
        self.feature_columns = metadata['feature_columns']
        self.params = metadata.get('params', self.params)
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True

''',
    'src/models/lightgbm_model.py': '''\"\"\"LightGBM specialist model for diversity.\"\"\"

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import joblib
from typing import Dict, Any

from .base_model import BaseModel, ModelSignal


class LightGBMModel(BaseModel):
    \"\"\"LightGBM model for price-based predictions (diversity model).\"\"\"
    
    def __init__(self, min_confidence: float = 0.6, **lgb_params):
        super().__init__("LightGBM", min_confidence)
        
        # Default LightGBM parameters
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            **lgb_params
        }
        
        self.model = None
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        \"\"\"Train LightGBM model.\"\"\"
        # Convert y to binary (0/1)
        y_binary = (y > 0).astype(int)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = ((train_pred > 0.5) == y_train).mean()
        val_acc = ((val_pred > 0.5) == y_val).mean()
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'n_features': len(self.feature_columns),
            'n_estimators': self.model.num_trees()
        }
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        \"\"\"Get prediction for a single snapshot.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X = pd.DataFrame([snapshot[self.feature_columns]])
        
        prob_up = float(self.model.predict(X)[0])
        confidence = self._calculate_confidence(prob_up)
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'prob_up': prob_up}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Predict on a batch.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X_aligned = X[self.feature_columns]
        
        prob_up = self.model.predict(X_aligned)
        confidence = np.array([self._calculate_confidence(p) for p in prob_up])
        signal = np.array([self._apply_abstention(p, c) 
                          for p, c in zip(prob_up, confidence)])
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        \"\"\"Save model and metadata.\"\"\"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'params': self.params,
            'min_confidence': self.min_confidence
        }
        joblib.dump(metadata, str(path).replace('.txt', '_metadata.pkl'))
    
    def load(self, filepath: str):
        \"\"\"Load model and metadata.\"\"\"
        path = Path(filepath)
        
        # Load model
        self.model = lgb.Booster(model_file=str(path))
        
        # Load metadata
        metadata = joblib.load(str(path).replace('.txt', '_metadata.pkl'))
        self.feature_columns = metadata['feature_columns']
        self.params = metadata.get('params', self.params)
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True


''',
    'src/models/sentiment_model.py': '''\"\"\"Sentiment-based prediction model using FinBERT/FinGPT.\"\"\"

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any, Optional

from .base_model import BaseModel, ModelSignal


class SentimentModel(BaseModel):
    \"\"\"Sentiment model using news data and pretrained transformers.\"\"\"
    
    def __init__(self, min_confidence: float = 0.6, use_pretrained: bool = True):
        super().__init__("Sentiment", min_confidence)
        self.use_pretrained = use_pretrained
        self.sentiment_model = None
        self.tokenizer = None
        self.feature_columns = None
        
        # Initialize transformer if requested
        if use_pretrained:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                # Use FinBERT (free, works on CPU)
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.sentiment_model.eval()  # Set to eval mode
                
                # Use CPU for free tier
                self.device = torch.device('cpu')
                self.sentiment_model.to(self.device)
                
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"Warning: Could not load FinBERT: {e}")
                print("Falling back to simple sentiment")
                self.use_pretrained = False
    
    def _get_sentiment_score(self, text: str) -> float:
        \"\"\"Get sentiment score for text.\"\"\"
        if pd.isna(text) or not text:
            return 0.0
        
        if self.use_pretrained and self.sentiment_model is not None:
            return self._get_finbert_sentiment(text)
        else:
            return self._get_simple_sentiment(text)
    
    def _get_finbert_sentiment(self, text: str) -> float:
        \"\"\"Get sentiment using FinBERT.\"\"\"
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, max_length=512,
                                  padding=True).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: positive, negative, neutral
            # Map to [-1, 1] scale
            positive_prob = probs[0][0].item()
            negative_prob = probs[0][1].item()
            neutral_prob = probs[0][2].item()
            
            # Return sentiment score: positive - negative
            sentiment = positive_prob - negative_prob
            return sentiment
            
        except Exception as e:
            print(f"Error in FinBERT prediction: {e}")
            return self._get_simple_sentiment(text)
    
    def _get_simple_sentiment(self, text: str) -> float:
        \"\"\"Simple keyword-based sentiment fallback.\"\"\"
        text_lower = str(text).lower()
        
        positive_words = ['up', 'gain', 'rise', 'surge', 'rally', 'bullish', 
                         'growth', 'profit', 'beat', 'strong', 'positive', 'buy']
        negative_words = ['down', 'fall', 'drop', 'decline', 'crash', 'bearish',
                         'loss', 'miss', 'weak', 'negative', 'warn', 'sell']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count + 1)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              news_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        \"\"\"
        Train sentiment model.
        
        Args:
            X: Feature matrix (should include news features)
            y: Target variable
            news_data: Optional raw news DataFrame for training
        \"\"\"
        # For sentiment model, we primarily use news features
        # Store which columns are news-related
        news_cols = [col for col in X.columns if 'news' in col.lower() or 'sentiment' in col.lower()]
        self.feature_columns = news_cols if news_cols else X.columns.tolist()
        
        # Simple training: learn weights for news features
        # In practice, this could be more sophisticated
        if len(news_cols) > 0:
            # Use news sentiment features directly
            X_sentiment = X[news_cols].fillna(0)
            
            # Simple linear combination
            # Weight by correlation with target
            correlations = X_sentiment.apply(lambda col: col.corr(y))
            self.feature_weights = correlations.fillna(0).to_dict()
        else:
            # Fallback: use all features
            self.feature_weights = {col: 1.0 / len(X.columns) for col in X.columns}
        
        self.is_trained = True
        
        # Calculate baseline accuracy
        if len(news_cols) > 0:
            X_sentiment = X[news_cols].fillna(0)
            weighted_sentiment = sum(X_sentiment[col] * self.feature_weights.get(col, 0) 
                                    for col in news_cols)
            predictions = (weighted_sentiment > 0).astype(int)
            accuracy = (predictions == (y > 0).astype(int)).mean()
        else:
            accuracy = 0.5  # Random baseline
        
        return {
            'train_accuracy': float(accuracy),
            'n_news_features': len(news_cols),
            'use_pretrained': self.use_pretrained
        }
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        \"\"\"Get prediction from sentiment features.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get sentiment score from news features
        sentiment_score = 0.0
        
        for col, weight in self.feature_weights.items():
            if col in snapshot:
                sentiment_score += snapshot[col] * weight
        
        # Convert sentiment score [-1, 1] to prob_up [0, 1]
        prob_up = (sentiment_score + 1) / 2
        prob_up = np.clip(prob_up, 0.0, 1.0)
        
        confidence = self._calculate_confidence(prob_up)
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'sentiment_score': sentiment_score}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Predict on a batch.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Calculate weighted sentiment
        sentiment_scores = np.zeros(len(X))
        
        for col, weight in self.feature_weights.items():
            if col in X.columns:
                sentiment_scores += X[col].fillna(0) * weight
        
        # Convert to prob_up
        prob_up = (sentiment_scores + 1) / 2
        prob_up = np.clip(prob_up, 0.0, 1.0)
        
        confidence = np.array([self._calculate_confidence(p) for p in prob_up])
        signal = np.array([self._apply_abstention(p, c) 
                          for p, c in zip(prob_up, confidence)])
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        \"\"\"Save model.\"\"\"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_columns': self.feature_columns,
            'feature_weights': self.feature_weights,
            'min_confidence': self.min_confidence,
            'use_pretrained': self.use_pretrained
        }
        joblib.dump(metadata, str(path))
    
    def load(self, filepath: str):
        \"\"\"Load model.\"\"\"
        path = Path(filepath)
        metadata = joblib.load(str(path))
        
        self.feature_columns = metadata['feature_columns']
        self.feature_weights = metadata['feature_weights']
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        self.use_pretrained = metadata.get('use_pretrained', False)
        
        self.is_trained = True


''',
    'src/models/rule_based_model.py': '''\"\"\"Simple rule-based baseline model.\"\"\"

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any

from .base_model import BaseModel, ModelSignal


class RuleBasedModel(BaseModel):
    \"\"\"Simple rule-based model for baseline comparison.\"\"\"
    
    def __init__(self, min_confidence: float = 0.6):
        super().__init__("RuleBased", min_confidence)
        self.rules = {}
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        \"\"\"Train rule-based model (learn simple thresholds).\"\"\"
        self.feature_columns = X.columns.tolist()
        
        # Learn simple rules based on feature correlations
        # Rule 1: Momentum-based
        momentum_cols = [col for col in X.columns if 'momentum' in col.lower() or 'return' in col.lower()]
        if momentum_cols:
            best_momentum = max(momentum_cols, key=lambda col: abs(X[col].corr(y)))
            self.rules['momentum_col'] = best_momentum
            self.rules['momentum_threshold'] = X[best_momentum].quantile(0.6)
        
        # Rule 2: RSI-based
        rsi_cols = [col for col in X.columns if 'rsi' in col.lower()]
        if rsi_cols:
            best_rsi = max(rsi_cols, key=lambda col: abs(X[col].corr(y)))
            self.rules['rsi_col'] = best_rsi
            self.rules['rsi_oversold'] = X[best_rsi].quantile(0.2)  # Buy when oversold
            self.rules['rsi_overbought'] = X[best_rsi].quantile(0.8)  # Sell when overbought
        
        # Rule 3: Moving average crossover
        ma_cols = [col for col in X.columns if 'ma_' in col.lower() and '_ratio' in col.lower()]
        if ma_cols:
            best_ma = max(ma_cols, key=lambda col: abs(X[col].corr(y)))
            self.rules['ma_col'] = best_ma
            self.rules['ma_threshold'] = 1.0  # Above MA = bullish
        
        self.is_trained = True
        
        # Calculate baseline accuracy
        predictions = self._apply_rules(X)
        accuracy = (predictions == (y > 0).astype(int)).mean()
        
        return {
            'train_accuracy': float(accuracy),
            'n_rules': len(self.rules),
            'n_features': len(self.feature_columns)
        }
    
    def _apply_rules(self, X: pd.DataFrame) -> np.ndarray:
        \"\"\"Apply learned rules to get predictions.\"\"\"
        predictions = np.zeros(len(X))
        
        # Rule 1: Momentum
        if 'momentum_col' in self.rules:
            col = self.rules['momentum_col']
            threshold = self.rules['momentum_threshold']
            if col in X.columns:
                predictions += (X[col] > threshold).astype(int) * 0.4
        
        # Rule 2: RSI
        if 'rsi_col' in self.rules:
            col = self.rules['rsi_col']
            oversold = self.rules.get('rsi_oversold', 30)
            overbought = self.rules.get('rsi_overbought', 70)
            if col in X.columns:
                # Buy when oversold, sell when overbought
                buy_signal = (X[col] < oversold).astype(int) * 0.3
                sell_signal = (X[col] > overbought).astype(int) * -0.3
                predictions += buy_signal + sell_signal
        
        # Rule 3: Moving average
        if 'ma_col' in self.rules:
            col = self.rules['ma_col']
            threshold = self.rules.get('ma_threshold', 1.0)
            if col in X.columns:
                predictions += (X[col] > threshold).astype(int) * 0.3
        
        # Convert to binary
        return (predictions > 0).astype(int)
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        \"\"\"Get prediction from rules.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Apply rules
        X = pd.DataFrame([snapshot[self.feature_columns]])
        prediction = self._apply_rules(X)[0]
        
        # Convert to probability
        prob_up = float(prediction)
        confidence = 0.5  # Rule-based models have lower confidence
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'rule_prediction': prediction}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Predict on a batch.\"\"\"
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self._apply_rules(X)
        prob_up = predictions.astype(float)
        confidence = np.full(len(X), 0.5)  # Low confidence for rules
        signal = np.array([self._apply_abstention(p, c) 
                          for p, c in zip(prob_up, confidence)])
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        \"\"\"Save model.\"\"\"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_columns': self.feature_columns,
            'rules': self.rules,
            'min_confidence': self.min_confidence
        }
        joblib.dump(metadata, str(path))
    
    def load(self, filepath: str):
        \"\"\"Load model.\"\"\"
        path = Path(filepath)
        metadata = joblib.load(str(path))
        
        self.feature_columns = metadata['feature_columns']
        self.rules = metadata['rules']
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True


''',
    'src/ensemble/meta_ensemble.py': '''\"\"\"Meta-gating ensemble that combines specialist models.\"\"\"

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import joblib

# Robust imports
import sys
from pathlib import Path
try:
    from ..models.base_model import BaseModel, ModelSignal
    from ..utils.config import DEFAULT_MIN_CONFIDENCE
except ImportError:
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from models.base_model import BaseModel, ModelSignal
    from utils.config import DEFAULT_MIN_CONFIDENCE


class MetaEnsemble:
    \"\"\"Meta-model that learns which specialist to trust and when.\"\"\"
    
    def __init__(self, specialists: List[BaseModel], min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.specialists = specialists
        self.min_confidence = min_confidence
        self.meta_model = None
        self.is_trained = False
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              market_features: Optional[pd.DataFrame] = None,
              **kwargs) -> Dict[str, Any]:
        \"\"\"
        Train meta-model to combine specialist predictions.
        
        Args:
            X: Feature matrix
            y: Target variable
            market_features: Optional market regime features (volatility, etc.)
        \"\"\"
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Get predictions from all specialists
        specialist_preds = {}
        for specialist in self.specialists:
            if not specialist.is_trained:
                print(f"Warning: {specialist.name} not trained, skipping")
                continue
            
            try:
                preds = specialist.predict_batch(X)
                specialist_preds[specialist.name] = preds
            except Exception as e:
                print(f"Error getting predictions from {specialist.name}: {e}")
                continue
        
        if not specialist_preds:
            raise ValueError("No trained specialists available")
        
        # Build meta-features
        meta_features = []
        feature_names = []
        
        # Specialist outputs
        for name, preds in specialist_preds.items():
            meta_features.append(preds['prob_up'].values)
            meta_features.append(preds['confidence'].values)
            feature_names.extend([f'{name}_prob', f'{name}_conf'])
        
        # Market regime features
        if market_features is not None:
            # Add volatility regime
            if 'volatility' in market_features.columns:
                meta_features.append(market_features['volatility'].values)
                feature_names.append('market_volatility')
            
            # Add news intensity
            news_cols = [col for col in market_features.columns if 'news' in col.lower()]
            if news_cols:
                meta_features.append(market_features[news_cols[0]].values)
                feature_names.append('news_intensity')
        
        # Combine meta-features
        meta_X = np.column_stack(meta_features)
        self.feature_columns = feature_names
        
        # Target: did the ensemble prediction match actual direction?
        y_binary = (y > 0).astype(int)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            meta_X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train meta-model (Random Forest for interpretability)
        self.meta_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.meta_model.predict(X_train)
        val_pred = self.meta_model.predict(X_val)
        
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()
        
        self.is_trained = True
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'n_specialists': len(specialist_preds),
            'n_meta_features': len(feature_names)
        }
    
    def get_signal(self, snapshot: pd.Series, 
                   market_snapshot: Optional[pd.Series] = None) -> ModelSignal:
        \"\"\"
        Get ensemble prediction for a single snapshot.
        
        Args:
            snapshot: Feature snapshot
            market_snapshot: Optional market regime features
        
        Returns:
            Combined ModelSignal with abstention
        \"\"\"
        if not self.is_trained:
            raise ValueError("Meta-ensemble not trained yet")
        
        # Get predictions from all specialists
        specialist_signals = []
        for specialist in self.specialists:
            if not specialist.is_trained:
                continue
            
            try:
                signal = specialist.get_signal(snapshot)
                specialist_signals.append((specialist.name, signal))
            except Exception as e:
                print(f"Error from {specialist.name}: {e}")
                continue
        
        if not specialist_signals:
            # Fallback: return abstention
            return ModelSignal(signal=0, prob_up=0.5, confidence=0.0)
        
        # Build meta-features
        meta_features = []
        for name, signal in specialist_signals:
            meta_features.append(signal.prob_up)
            meta_features.append(signal.confidence)
        
        # Add market features if available
        if market_snapshot is not None:
            if 'volatility' in market_snapshot.index:
                meta_features.append(market_snapshot['volatility'])
            news_cols = [col for col in market_snapshot.index if 'news' in col.lower()]
            if news_cols:
                meta_features.append(market_snapshot[news_cols[0]])
        
        # Get meta-model prediction
        meta_X = np.array([meta_features])
        prob_up = self.meta_model.predict_proba(meta_X)[0][1]
        
        # Calculate confidence (weighted average of specialist confidences)
        confidences = [s.confidence for _, s in specialist_signals]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Apply abstention
        if avg_confidence < self.min_confidence:
            signal = 0  # Abstain
        elif prob_up > 0.5:
            signal = 1  # Buy
        else:
            signal = -1  # Sell
        
        return ModelSignal(
            signal=signal,
            prob_up=float(prob_up),
            confidence=float(avg_confidence),
            raw_output={
                'specialist_signals': {name: s.signal for name, s in specialist_signals},
                'meta_prob': prob_up
            }
        )
    
    def predict_batch(self, X: pd.DataFrame,
                     market_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        \"\"\"Predict on a batch.\"\"\"
        if not self.is_trained:
            raise ValueError("Meta-ensemble not trained yet")
        
        # Get predictions from all specialists
        specialist_preds = {}
        for specialist in self.specialists:
            if not specialist.is_trained:
                continue
            
            try:
                preds = specialist.predict_batch(X)
                specialist_preds[specialist.name] = preds
            except Exception as e:
                print(f"Error from {specialist.name}: {e}")
                continue
        
        if not specialist_preds:
            # Return abstention for all
            return pd.DataFrame({
                'signal': 0,
                'prob_up': 0.5,
                'confidence': 0.0
            }, index=X.index)
        
        # Build meta-features
        meta_features_list = []
        for name, preds in specialist_preds.items():
            meta_features_list.append(preds['prob_up'].values)
            meta_features_list.append(preds['confidence'].values)
        
        # Add market features if available
        if market_features is not None:
            if 'volatility' in market_features.columns:
                meta_features_list.append(market_features['volatility'].values)
            news_cols = [col for col in market_features.columns if 'news' in col.lower()]
            if news_cols:
                meta_features_list.append(market_features[news_cols[0]].values)
        
        # Get meta-model predictions
        meta_X = np.column_stack(meta_features_list)
        prob_up = self.meta_model.predict_proba(meta_X)[:, 1]
        
        # Calculate confidence (average of specialist confidences)
        confidences = [preds['confidence'].values for preds in specialist_preds.values()]
        avg_confidence = np.mean(confidences, axis=0)
        
        # Apply abstention
        signal = np.where(avg_confidence < self.min_confidence, 0,
                         np.where(prob_up > 0.5, 1, -1))
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': avg_confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        \"\"\"Save meta-ensemble.\"\"\"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_columns': self.feature_columns,
            'min_confidence': self.min_confidence,
            'specialist_names': [s.name for s in self.specialists]
        }
        
        # Save meta-model
        joblib.dump(self.meta_model, str(path))
        
        # Save metadata
        joblib.dump(metadata, str(path).replace('.pkl', '_metadata.pkl'))
    
    def load(self, filepath: str):
        \"\"\"Load meta-ensemble.\"\"\"
        path = Path(filepath)
        
        # Load meta-model
        self.meta_model = joblib.load(str(path))
        
        # Load metadata
        metadata = joblib.load(str(path).replace('.pkl', '_metadata.pkl'))
        self.feature_columns = metadata['feature_columns']
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True

''',
    'src/backtest/walkforward_backtest.py': '''\"\"\"Walk-forward backtesting framework.\"\"\"

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Robust imports
import sys
from pathlib import Path
try:
    from ..ensemble.meta_ensemble import MetaEnsemble
    from ..utils.config import DEFAULT_TRAIN_WINDOW_DAYS, DEFAULT_TEST_WINDOW_DAYS, INITIAL_CAPITAL, COMMISSION_RATE
    from ..utils.helpers import ensure_no_leakage, save_artifact
except ImportError:
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from ensemble.meta_ensemble import MetaEnsemble
    from utils.config import DEFAULT_TRAIN_WINDOW_DAYS, DEFAULT_TEST_WINDOW_DAYS, INITIAL_CAPITAL, COMMISSION_RATE
    from utils.helpers import ensure_no_leakage, save_artifact


class WalkForwardBacktest:
    \"\"\"Walk-forward backtesting with rolling windows.\"\"\"
    
    def __init__(self, ensemble: MetaEnsemble, 
                 train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
                 test_window_days: int = DEFAULT_TEST_WINDOW_DAYS,
                 initial_capital: float = INITIAL_CAPITAL,
                 commission_rate: float = COMMISSION_RATE):
        self.ensemble = ensemble
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        self.results = []
    
    def run(self, features: pd.DataFrame, prices: pd.DataFrame,
            target: pd.Series) -> Dict[str, Any]:
        \"\"\"
        Run walk-forward backtest.
        
        Args:
            features: Feature matrix (indexed by date)
            prices: Price data with 'close' column
            target: Target returns (forward-looking)
        
        Returns:
            Dictionary with backtest results
        \"\"\"
        # Ensure dates are sorted
        features = features.sort_index()
        prices = prices.sort_index()
        target = target.sort_index()
        
        # Align all data
        common_dates = features.index.intersection(prices.index).intersection(target.index)
        features = features.loc[common_dates]
        prices = prices.loc[common_dates]
        target = target.loc[common_dates]
        
        # Calculate windows
        start_date = features.index.min()
        end_date = features.index.max()
        
        current_date = start_date + pd.Timedelta(days=self.train_window_days)
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        window_num = 0
        
        while current_date + pd.Timedelta(days=self.test_window_days) <= end_date:
            window_num += 1
            
            # Define train and test windows
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.Timedelta(days=self.test_window_days)
            
            train_mask = (features.index >= start_date) & (features.index < train_end)
            test_mask = (features.index >= test_start) & (features.index < test_end)
            
            train_features = features[train_mask]
            test_features = features[test_mask]
            test_target = target[test_mask]
            test_prices = prices.loc[prices.index.isin(test_features.index)]
            
            if len(train_features) < 50 or len(test_features) == 0:
                current_date += pd.Timedelta(days=self.test_window_days)
                continue
            
            print(f"\\nWindow {window_num}: Train {train_features.index.min()} to {train_features.index.max()}, "
                  f"Test {test_features.index.min()} to {test_features.index.max()}")
            
            # Train ensemble (specialists should already be trained, but retrain meta-model)
            try:
                # Get market features for meta-training
                market_features = self._extract_market_features(train_features)
                
                # Train meta-model on training window
                train_target = target[train_mask]
                self.ensemble.train(train_features, train_target, market_features)
                
                # Predict on test window
                test_market_features = self._extract_market_features(test_features)
                predictions = self.ensemble.predict_batch(test_features, test_market_features)
                
                # Store results
                for idx, (date, pred_row) in enumerate(predictions.iterrows()):
                    if idx < len(test_target):
                        all_predictions.append({
                            'date': date,
                            'signal': pred_row['signal'],
                            'prob_up': pred_row['prob_up'],
                            'confidence': pred_row['confidence']
                        })
                        all_actuals.append({
                            'date': date,
                            'actual_return': test_target.iloc[idx] if idx < len(test_target) else 0,
                            'actual_direction': 1 if test_target.iloc[idx] > 0 else -1 if idx < len(test_target) else 0
                        })
                        all_dates.append(date)
                
            except Exception as e:
                print(f"Error in window {window_num}: {e}")
                continue
            
            # Move window forward
            current_date += pd.Timedelta(days=self.test_window_days)
        
        # Calculate metrics
        if not all_predictions:
            return {'error': 'No predictions generated'}
        
        results_df = pd.DataFrame(all_predictions).set_index('date')
        actuals_df = pd.DataFrame(all_actuals).set_index('date')
        
        # Align
        common_dates = results_df.index.intersection(actuals_df.index)
        results_df = results_df.loc[common_dates]
        actuals_df = actuals_df.loc[common_dates]
        
        # Calculate metrics
        metrics = self._calculate_metrics(results_df, actuals_df, prices.loc[common_dates])
        
        # Store results
        self.results = {
            'predictions': results_df,
            'actuals': actuals_df,
            'metrics': metrics,
            'n_windows': window_num
        }
        
        return self.results
    
    def _extract_market_features(self, features: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Extract market regime features.\"\"\"
        market_features = pd.DataFrame(index=features.index)
        
        # Volatility
        vol_cols = [col for col in features.columns if 'volatility' in col.lower()]
        if vol_cols:
            market_features['volatility'] = features[vol_cols[0]]
        else:
            market_features['volatility'] = 0.0
        
        # News intensity
        news_cols = [col for col in features.columns if 'news_count' in col.lower()]
        if news_cols:
            market_features['news_intensity'] = features[news_cols[0]]
        else:
            market_features['news_intensity'] = 0.0
        
        return market_features.fillna(0)
    
    def _calculate_metrics(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                          prices: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Calculate backtest metrics.\"\"\"
        # Directional accuracy
        predicted_direction = np.where(predictions['signal'] != 0, 
                                      np.where(predictions['signal'] > 0, 1, -1), 
                                      0)
        actual_direction = actuals['actual_direction'].values
        
        # Only count non-abstained predictions
        non_abstain_mask = predictions['signal'] != 0
        if non_abstain_mask.sum() > 0:
            directional_accuracy = (predicted_direction[non_abstain_mask] == 
                                  actual_direction[non_abstain_mask]).mean()
        else:
            directional_accuracy = 0.0
        
        # Coverage (percentage of days traded)
        coverage = non_abstain_mask.mean()
        
        # Precision and recall
        buy_signals = predictions['signal'] == 1
        sell_signals = predictions['signal'] == -1
        
        if buy_signals.sum() > 0:
            buy_precision = (actual_direction[buy_signals] == 1).mean()
            buy_recall = (actual_direction[buy_signals] == 1).sum() / (actual_direction == 1).sum() if (actual_direction == 1).sum() > 0 else 0
        else:
            buy_precision = buy_recall = 0.0
        
        if sell_signals.sum() > 0:
            sell_precision = (actual_direction[sell_signals] == -1).mean()
            sell_recall = (actual_direction[sell_signals] == -1).sum() / (actual_direction == -1).sum() if (actual_direction == -1).sum() > 0 else 0
        else:
            sell_precision = sell_recall = 0.0
        
        # Confidence calibration
        if non_abstain_mask.sum() > 0:
            avg_confidence = predictions.loc[non_abstain_mask, 'confidence'].mean()
        else:
            avg_confidence = 0.0
        
        # PnL simulation
        pnl_results = self._simulate_pnl(predictions, actuals, prices)
        
        return {
            'directional_accuracy': float(directional_accuracy),
            'coverage': float(coverage),
            'buy_precision': float(buy_precision),
            'buy_recall': float(buy_recall),
            'sell_precision': float(sell_precision),
            'sell_recall': float(sell_recall),
            'avg_confidence': float(avg_confidence),
            'total_return': float(pnl_results['total_return']),
            'sharpe_ratio': float(pnl_results['sharpe_ratio']),
            'max_drawdown': float(pnl_results['max_drawdown']),
            'win_rate': float(pnl_results['win_rate'])
        }
    
    def _simulate_pnl(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                     prices: pd.DataFrame) -> Dict[str, float]:
        \"\"\"Simulate PnL from predictions.\"\"\"
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        
        returns = []
        equity_curve = [capital]
        
        for date in predictions.index:
            if date not in prices.index:
                continue
            
            signal = predictions.loc[date, 'signal']
            current_price = prices.loc[date, 'close']
            actual_return = actuals.loc[date, 'actual_return']
            
            # Close position if signal changes
            if position != 0 and signal != position:
                # Close position
                if position == 1:  # Close long
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # Close short
                    pnl_pct = (entry_price - current_price) / entry_price
                
                capital *= (1 + pnl_pct - self.commission_rate)
                returns.append(pnl_pct)
                position = 0
            
            # Open new position
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
            
            equity_curve.append(capital)
        
        # Close final position
        if position != 0 and len(prices) > 0:
            final_price = prices.iloc[-1]['close']
            if position == 1:
                pnl_pct = (final_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - final_price) / entry_price
            capital *= (1 + pnl_pct - self.commission_rate)
            returns.append(pnl_pct)
        
        # Calculate metrics
        if returns:
            total_return = (capital / self.initial_capital) - 1
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            win_rate = (np.array(returns) > 0).mean()
            
            # Max drawdown
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = abs(drawdown.min())
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_capital': capital
        }
    
    def save_results(self, filepath: str):
        \"\"\"Save backtest results.\"\"\"
        if not self.results:
            raise ValueError("No results to save")
        
        save_artifact(self.results, filepath, format='pickle')

''',
    'src/data/__init__.py': '''from .price_fetcher import PriceFetcher
from .news_fetcher import NewsFetcher
__all__ = ["PriceFetcher", "NewsFetcher"]''',
    'src/features/__init__.py': '''from .feature_builder import FeatureBuilder
__all__ = ["FeatureBuilder"]''',
    'src/models/__init__.py': '''from .base_model import BaseModel, ModelSignal
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .sentiment_model import SentimentModel
from .rule_based_model import RuleBasedModel
__all__ = ["BaseModel", "ModelSignal", "XGBoostModel", "LightGBMModel", "SentimentModel", "RuleBasedModel"]''',
    'src/ensemble/__init__.py': '''from .meta_ensemble import MetaEnsemble
__all__ = ["MetaEnsemble"]''',
    'src/backtest/__init__.py': '''from .walkforward_backtest import WalkForwardBacktest
__all__ = ["WalkForwardBacktest"]''',
}

# Write all files
print("Creating source files...")
for file_path, content in files_content.items():
    full_path = project_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    print(f"  ✓ Created {file_path}")

print(f"\n✓ All source files created!")
print(f"   Project ready at: {project_dir}")
