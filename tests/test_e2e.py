"""End-to-end test on small dataset."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add src to path and import
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import PriceFetcher
from features import FeatureBuilder
from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel
from ensemble import MetaEnsemble
from backtest import WalkForwardBacktest


def test_e2e_pipeline():
    """Test complete pipeline on synthetic data."""
    print("\n" + "="*60)
    print("END-TO-END TEST")
    print("="*60)
    
    # Step 1: Create synthetic price data
    print("\n[1/6] Creating synthetic data...")
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Generate realistic price series
    returns = np.random.randn(200) * 0.02
    prices_array = 100 * (1 + returns).cumprod()
    
    prices = pd.DataFrame({
        'open': prices_array * (1 + np.random.randn(200) * 0.001),
        'high': prices_array * (1 + abs(np.random.randn(200)) * 0.01),
        'low': prices_array * (1 - abs(np.random.randn(200)) * 0.01),
        'close': prices_array,
        'volume': np.random.randint(1000000, 10000000, 200)
    }, index=dates)
    
    # Ensure OHLC consistency
    prices['high'] = prices[['high', 'open', 'close']].max(axis=1)
    prices['low'] = prices[['low', 'open', 'close']].min(axis=1)
    
    # Create index data
    index_returns = np.random.randn(200) * 0.015
    index_prices = pd.DataFrame({
        'close': 3000 * (1 + index_returns).cumprod()
    }, index=dates)
    
    print(f"✓ Created {len(prices)} days of price data")
    
    # Step 2: Build features
    print("\n[2/6] Building features...")
    builder = FeatureBuilder()
    features = builder.build_all_features(prices, index_prices)
    print(f"✓ Built {len(features.columns)} features, {len(features)} samples")
    
    # Step 3: Prepare training data
    print("\n[3/6] Preparing training data...")
    feature_cols = [col for col in features.columns 
                   if col not in ['target_return_1d', 'target_direction', 
                                 'open', 'high', 'low', 'close', 'volume']]
    X = features[feature_cols].fillna(0)
    y = features['target_direction']
    
    # Remove zero targets
    mask = y != 0
    X = X[mask]
    y = y[mask]
    
    # Split train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✓ Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Step 4: Train specialist models
    print("\n[4/6] Training specialist models...")
    
    xgb = XGBoostModel(min_confidence=0.5, n_estimators=20)  # Small for speed
    xgb.train(X_train, y_train)
    print(f"  ✓ XGBoost: {xgb.is_trained}")
    
    lgb = LightGBMModel(min_confidence=0.5)
    lgb.params['num_boost_round'] = 20
    lgb.train(X_train, y_train)
    print(f"  ✓ LightGBM: {lgb.is_trained}")
    
    sentiment = SentimentModel(min_confidence=0.5, use_pretrained=False)
    sentiment.train(X_train, y_train)
    print(f"  ✓ Sentiment: {sentiment.is_trained}")
    
    rule = RuleBasedModel(min_confidence=0.5)
    rule.train(X_train, y_train)
    print(f"  ✓ Rule-based: {rule.is_trained}")
    
    specialists = [xgb, lgb, sentiment, rule]
    
    # Step 5: Train meta-ensemble
    print("\n[5/6] Training meta-ensemble...")
    
    # Extract market features
    vol_cols = [col for col in X_train.columns if 'volatility' in col.lower()]
    news_cols = [col for col in X_train.columns if 'news' in col.lower()]
    
    market_train = pd.DataFrame(index=X_train.index)
    if vol_cols:
        market_train['volatility'] = X_train[vol_cols[0]]
    if news_cols:
        market_train['news_intensity'] = X_train[news_cols[0]]
    market_train = market_train.fillna(0)
    
    ensemble = MetaEnsemble(specialists, min_confidence=0.5)
    meta_metrics = ensemble.train(X_train, y_train, market_train)
    print(f"  ✓ Meta-ensemble trained: {meta_metrics}")
    
    # Step 6: Test predictions
    print("\n[6/6] Testing predictions...")
    
    # Get predictions on test set
    market_test = pd.DataFrame(index=X_test.index)
    if vol_cols:
        market_test['volatility'] = X_test[vol_cols[0]]
    if news_cols:
        market_test['news_intensity'] = X_test[news_cols[0]]
    market_test = market_test.fillna(0)
    
    predictions = ensemble.predict_batch(X_test, market_test)
    
    # Calculate accuracy
    non_abstain = predictions['signal'] != 0
    if non_abstain.sum() > 0:
        pred_dir = np.where(predictions.loc[non_abstain, 'signal'] > 0, 1, -1)
        actual_dir = y_test.loc[non_abstain].values
        accuracy = (pred_dir == actual_dir).mean()
        coverage = non_abstain.mean()
        
        print(f"  ✓ Accuracy: {accuracy:.3f}")
        print(f"  ✓ Coverage: {coverage:.3f}")
        
        assert accuracy > 0.4, "Accuracy too low (should be > 0.4 for random baseline)"
        assert coverage > 0, "No predictions made"
    else:
        print("  ⚠ All predictions abstained (low confidence)")
    
    print("\n" + "="*60)
    print("END-TO-END TEST PASSED")
    print("="*60)
    
    return True


if __name__ == "__main__":
    test_e2e_pipeline()

