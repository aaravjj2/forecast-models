"""Test specialist models."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel


def test_xgboost_model():
    """Test XGBoost model."""
    # Create sample data
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.choice([-1, 1], 100))
    
    model = XGBoostModel(min_confidence=0.6)
    metrics = model.train(X, y)
    
    assert model.is_trained
    assert 'train_accuracy' in metrics
    
    # Test prediction
    signal = model.get_signal(X.iloc[0])
    assert signal.signal in [-1, 0, 1]
    assert 0 <= signal.prob_up <= 1
    assert 0 <= signal.confidence <= 1


def test_lightgbm_model():
    """Test LightGBM model."""
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.choice([-1, 1], 100))
    
    model = LightGBMModel(min_confidence=0.6)
    metrics = model.train(X, y)
    
    assert model.is_trained
    assert 'train_accuracy' in metrics
    
    signal = model.get_signal(X.iloc[0])
    assert signal.signal in [-1, 0, 1]


def test_sentiment_model():
    """Test sentiment model."""
    X = pd.DataFrame({
        'news_sentiment_7d': np.random.randn(100),
        'news_count_7d': np.random.randint(0, 10, 100)
    })
    y = pd.Series(np.random.choice([-1, 1], 100))
    
    model = SentimentModel(min_confidence=0.6, use_pretrained=False)
    metrics = model.train(X, y)
    
    assert model.is_trained
    
    signal = model.get_signal(X.iloc[0])
    assert signal.signal in [-1, 0, 1]


def test_rule_based_model():
    """Test rule-based model."""
    X = pd.DataFrame({
        'momentum_5d': np.random.randn(100),
        'rsi_14': np.random.uniform(0, 100, 100),
        'ma_20_ratio': np.random.uniform(0.9, 1.1, 100)
    })
    y = pd.Series(np.random.choice([-1, 1], 100))
    
    model = RuleBasedModel(min_confidence=0.6)
    metrics = model.train(X, y)
    
    assert model.is_trained
    
    signal = model.get_signal(X.iloc[0])
    assert signal.signal in [-1, 0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

