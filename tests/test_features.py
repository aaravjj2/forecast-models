"""Test feature engineering."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import FeatureBuilder


def test_feature_builder():
    """Test feature builder creates features correctly."""
    # Create sample price data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Ensure high >= close >= low
    prices['high'] = prices[['high', 'close']].max(axis=1)
    prices['low'] = prices[['low', 'close']].min(axis=1)
    
    builder = FeatureBuilder()
    features = builder.build_price_features(prices)
    
    # Check features were created
    assert len(features.columns) > len(prices.columns)
    assert 'return_1d' in features.columns
    assert 'rsi_14' in features.columns
    assert 'volatility_5d' in features.columns
    
    # Check no NaN in final features (after dropna)
    features_clean = features.dropna()
    assert not features_clean.empty


def test_feature_builder_with_index():
    """Test feature builder with index data."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    prices['high'] = prices[['high', 'close']].max(axis=1)
    prices['low'] = prices[['low', 'close']].min(axis=1)
    
    index_prices = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 3000
    }, index=dates)
    
    builder = FeatureBuilder()
    features = builder.build_market_features(prices, index_prices)
    
    assert 'index_return_1d' in features.columns
    assert 'relative_strength_5d' in features.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

