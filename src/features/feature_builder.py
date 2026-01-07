"""Build features from price and news data."""

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
    """Build time-aligned, leak-free features from price and news data."""
    
    def __init__(self):
        self.feature_metadata = {}
    
    def build_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build price-based features.
        
        Args:
            prices: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            DataFrame with engineered features
        """
        df = prices.copy()
        
        # Returns over multiple periods
        for period in LOOKBACK_WINDOWS:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Volatility measures & Term Structure (B1)
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std()
            df[f'realized_vol_{period}d'] = df[f'volatility_{period}d'] * np.sqrt(252)
            
        # Volatility Term Structure (Short / Long)
        df['vol_term_structure'] = safe_divide(df['volatility_5d'], df['volatility_60d'])
        
        # [MOVED DOWN] Intraday Range & Gap (B1)

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
        
        # Intraday Range & Gap (B1) - Moved here to use ATR
        df['intraday_range_pct'] = (df['high'] - df['low']) / df['open']
        df['overnight_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        # Gap relative to ATR
        df['gap_vs_atr'] = safe_divide(df['overnight_gap_pct'].abs(), df['atr_14'])
        
        # Price position in range
        for window in [20, 50]:
            df[f'high_{window}'] = df['high'].rolling(window).max()
            df[f'low_{window}'] = df['low'].rolling(window).min()
            df[f'price_position_{window}'] = safe_divide(
                df['close'] - df[f'low_{window}'],
                df[f'high_{window}'] - df[f'low_{window}']
            )
        
        # Volume & Liquidity Features (B2)
        for window in [10, 20, 50]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            # Volume Shock (Ratio)
            df[f'volume_ratio_{window}'] = safe_divide(df['volume'], df[f'volume_ma_{window}'])
        
        # Amihud Illiquidity Proxy (AbsRet / DollarVolume)
        # Dollar Volume = Close * Volume
        dollar_vol = df['close'] * df['volume']
        df['amihud_illiquidity'] = safe_divide(df['return_1d'].abs(), dollar_vol).rolling(20).mean() * 1e6 # Scale up
        
        # Stationary Amihud (Relative to recent history) - Essential for Liquidity Regime detection
        df['amihud_ma_60'] = df['amihud_illiquidity'].rolling(60).mean()
        df['amihud_rel_60'] = safe_divide(df['amihud_illiquidity'], df['amihud_ma_60'])
        
        # Price Impact Asymmetry (Upside Vol vs Downside Vol per unit volume)? 
        # Simpler: Correlation of Volume and Abs Return (if high, liquidity is thinner?)
        df['vol_ret_corr_20'] = df['volume'].rolling(20).corr(df['return_1d'].abs())
        
        # Price change features
        df['high_low_ratio'] = safe_divide(df['high'], df['low'])
        df['close_open_ratio'] = safe_divide(df['close'], df['open'])
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
        
        return df
    
    def build_market_features(self, prices: pd.DataFrame, 
                             index_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build market context features using index data.
        
        Args:
            prices: Stock price DataFrame
            index_prices: Index price DataFrame (e.g., S&P 500)
        
        Returns:
            DataFrame with market features
        """
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
        """
        Build news-based features.
        
        Args:
            prices: Stock price DataFrame (indexed by date)
            news: News DataFrame with columns: date, headline, summary
        
        Returns:
            DataFrame with news features
        """
        df = prices.copy()
        
        if news.empty:
            # Fill with zeros if no news
            df['news_count_1d'] = 0
            df['news_count_7d'] = 0
            df['news_sentiment_7d'] = 0
            return df
        
        # Normalize news dates to midnight for alignment
        news_aligned = news.copy()
        news_aligned['date'] = pd.to_datetime(news_aligned['date']).dt.normalize()
        news_aligned = news_aligned.set_index('date')
        
        # Reindex to price dates (use 'date' column or index depending on input)
        # Note: prices is indexed by date in build_all_features
        # Ensure prices index is also normalized if it contains times
        target_index = pd.to_datetime(df.index).normalize()
        
        # Group by date to handle multiple articles per day
        # We need to reindex against the TARGET index.
        # But first, we aggregate daily news.
        daily_news_count = news_aligned.groupby(news_aligned.index).size()
        
        # Now reindex to price dates to get daily series aligned with market days
        # fill_value=0 implies no news on that market day
        aligned_counts = daily_news_count.reindex(target_index, fill_value=0)
        
        # Sentiment aggregation
        if 'headline' in news.columns:
            news_aligned['simple_sentiment'] = news_aligned['headline'].apply(self._simple_sentiment)
            daily_sentiment = news_aligned.groupby(news_aligned.index)['simple_sentiment'].mean()
            aligned_sentiment = daily_sentiment.reindex(target_index, fill_value=0)
        else:
            aligned_sentiment = pd.Series(0, index=target_index)

        # STRICT LEAKAGE PREVENTION: Shift News Features by 1 Day
        # We assume we only have access to Yesterday's news for Today's Open trade decision.
        # Or more strictly: Open T+1 decision relies on data known to Close T.
        # If we use Close T, we know News T. 
        # BUT user explicitly requested: "Re-test with delayed sentiment (T+1)"
        # So we MUST shift(1).
        aligned_counts = aligned_counts.shift(1).fillna(0)
        aligned_sentiment = aligned_sentiment.shift(1).fillna(0)
        
        # Assign back to df (ensure index matches original df)
        aligned_counts.index = df.index
        aligned_sentiment.index = df.index
        
        # B3. Information Delay Signals: Divergence
        # Compare "Yesterday's News" (aligned_sentiment at row T) with "Yesterday's Price" (return_1d.shift(1) at row T)
        # return_1d at row T is Today's return (Close T / Close T-1)
        # return_1d.shift(1) is Yesterday's return (Close T-1 / Close T-2)
        # Normalize return by 20d volatility for scale
        vol_20d = df['return_1d'].rolling(20).std()
        normalized_past_return = safe_divide(df['return_1d'].shift(1), vol_20d)
        
        # Divergence: High Value = High News Sentiment but Low Price Reaction (Bullish Divergence/Underreaction)
        # Note: aligned_sentiment is already shifted(1) inside this df row context
        df['news_price_divergence'] = aligned_sentiment - normalized_past_return.fillna(0)
        
        # Calculate rolling features on the SHIFTED (T+1) series
        for window in [1, 3, 7, 14]:
            df[f'news_count_{window}d'] = aligned_counts.rolling(window, min_periods=0).sum()
            
        for window in [3, 7, 14]:
             df[f'news_sentiment_{window}d'] = aligned_sentiment.rolling(window, min_periods=0).mean()
        
        return df
    
    def build_all_features(self, prices: pd.DataFrame, 
                          index_prices: Optional[pd.DataFrame] = None,
                          news: Optional[pd.DataFrame] = None,
                          target_col: str = 'close') -> pd.DataFrame:
        """
        Build all features and create target variable.
        
        Args:
            prices: Price DataFrame
            index_prices: Optional index DataFrame
            news: Optional news DataFrame
            target_col: Column to use for target (default: 'close')
        
        Returns:
            DataFrame with all features and target
        """
        # Start with price features
        df = self.build_price_features(prices)
        
        # Add market features if available
        if index_prices is not None and not index_prices.empty:
            df = self.build_market_features(df, index_prices)
        
        # Add news features if available
        if news is not None and not news.empty:
            df = self.build_news_features(df, news)
        
        # Create target: Open-to-Open return (Decision at Close_t -> Buy Open_{t+1} -> Sell Open_{t+2})
        # Shift(-1) gives t+1 data. Shift(-2) gives t+2 data.
        # RENAME TO next_day_return to avoid usage as ML target
        df['next_day_return'] = (df['open'].shift(-2) - df['open'].shift(-1)) / df['open'].shift(-1)
        
        # Remove directional binary target (A1. BAN DIRECTIONAL PREDICTION)
        # df['target_direction'] = ... DELETED
        
        # Fill NaN values with 0 (from rolling calculations at start)
        # Only drop rows where target is NaN (can't backtest without target)
        df = df.fillna(0)
        df = df.dropna(subset=['next_day_return'])
        
        # Store feature metadata
        feature_cols = [col for col in df.columns 
                       if col not in ['next_day_return', 'target_return_1d', 'target_direction',
                                     'open', 'high', 'low', 'close', 'volume']]
        self.feature_metadata = {
            'feature_columns': feature_cols,
            'target_columns': ['next_day_return'], # Only for backtesting
            'n_features': len(feature_cols)
        }
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = safe_divide(gain, loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment (placeholder for FinBERT)."""
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

