#!/usr/bin/env python3
"""
Robustness & Validation Testing Suite for ML Pipeline

This script runs comprehensive tests to validate model performance:
- PHASE 1: Sanity & robustness checks
- PHASE 2: Expand universe (multi-stock)
- PHASE 3: Stress ensemble logic
- PHASE 4: Reality alignment
- PHASE 5: Decision framework
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup
project_root = Path(__file__).parent
env_file = project_root.parent / 'keys.env'
if env_file.exists():
    load_dotenv(env_file)

sys.path.insert(0, str(project_root / 'src'))

from data import PriceFetcher, NewsFetcher
from features import FeatureBuilder
from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel
from ensemble import MetaEnsemble
from backtest import WalkForwardBacktest
from utils.config import PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR, META_MODEL_DIR, RESULTS_DIR
from utils.helpers import save_artifact

# Create results directory
test_results_dir = project_root / 'robustness_test_results'
test_results_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ROBUSTNESS & VALIDATION TESTING SUITE")
print("="*70)
print(f"Results will be saved to: {test_results_dir}\n")


# ============================================================================
# PHASE 1: SANITY & ROBUSTNESS CHECKS
# ============================================================================

def test_confidence_thresholds(ticker: str = "AAPL", 
                                thresholds: List[float] = [0.7, 0.6, 0.55, 0.5, 0.45]) -> pd.DataFrame:
    """
    Test 1: Lower confidence threshold gradually
    Records: trades count, accuracy, Sharpe, max drawdown
    """
    print("\n" + "="*70)
    print("PHASE 1.1: CONFIDENCE THRESHOLD TEST")
    print("="*70)
    
    # Load data
    features = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_prices.csv", index_col=0, parse_dates=True)
    
    feature_cols = [col for col in features.columns 
                    if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
    X = features[feature_cols].fillna(0)
    y = features['target_return_1d']
    y_dir = features['target_direction']
    
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        
        # Load or train models with new threshold
        xgb = XGBoostModel(min_confidence=threshold)
        lgb = LightGBMModel(min_confidence=threshold)
        sentiment = SentimentModel(min_confidence=threshold, use_pretrained=False)
        rule = RuleBasedModel(min_confidence=threshold)
        
        # Train if needed
        mask = y_dir != 0
        X_train = X[mask]
        y_train = y_dir[mask]
        
        xgb.train(X_train, y_train)
        lgb.train(X_train, y_train)
        sentiment.train(X_train, y_train)
        rule.train(X_train, y_train)
        
        specialists = [xgb, lgb, sentiment, rule]
        # Market features (aligned with training data)
        vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
        news_cols = [col for col in X.columns if 'news' in col.lower()]
        market_features = pd.DataFrame(index=X[mask].index)
        if vol_cols and vol_cols[0] in X_train.columns:
            market_features['volatility'] = X_train[vol_cols[0]]
        else:
            market_features['volatility'] = 0
        if news_cols and news_cols[0] in X_train.columns:
            market_features['news_intensity'] = X_train[news_cols[0]]
        else:
            market_features['news_intensity'] = 0
        market_features = market_features.fillna(0)
        
        # Train ensemble
        ensemble = MetaEnsemble(specialists, min_confidence=threshold)
        ensemble.train(X_train, y_train, market_features)
        
        # Run backtest
        backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
        test_results = backtest.run(X, prices, y)
        
        metrics = test_results['metrics']
        predictions = test_results['predictions']
        
        # Count trades
        trades = (predictions['signal'] != 0).sum()
        
        results.append({
            'threshold': threshold,
            'trades': trades,
            'accuracy': float(metrics.get('directional_accuracy', 0)),
            'sharpe': float(metrics.get('sharpe_ratio', 0)),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'total_return': float(metrics.get('total_return', 0)),
            'coverage': metrics.get('coverage', 0),
            'win_rate': metrics.get('win_rate', 0)
        })
        
        print(f"  Trades: {trades}, Accuracy: {metrics.get('directional_accuracy', 0):.4f}, "
              f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(test_results_dir / f"{ticker}_confidence_threshold_test.csv", index=False)
    
    print(f"\n✓ Results saved to: {test_results_dir / f'{ticker}_confidence_threshold_test.csv'}")
    print("\nSummary:")
    print(df_results.to_string(index=False))
    
    return df_results


def test_shuffle_labels(ticker: str = "AAPL") -> Dict:
    """
    Test 2: Shuffle test (anti-overfitting)
    Randomly shuffle labels and re-run pipeline
    Expected: Accuracy ≈ 50%, no profitable trades
    """
    print("\n" + "="*70)
    print("PHASE 1.2: SHUFFLE TEST (Anti-Overfitting)")
    print("="*70)
    
    # Load data
    features = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_prices.csv", index_col=0, parse_dates=True)
    
    feature_cols = [col for col in features.columns 
                    if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
    X = features[feature_cols].fillna(0)
    y = features['target_return_1d']
    y_dir = features['target_direction'].copy()
    
    # SHUFFLE LABELS
    print("⚠️  Shuffling labels (destroying signal)...")
    np.random.seed(42)
    y_dir_shuffled = y_dir.copy()
    mask = y_dir_shuffled != 0
    y_dir_shuffled[mask] = np.random.permutation(y_dir_shuffled[mask].values)
    
    # Train models with shuffled labels
    print("Training models with shuffled labels...")
    mask = y_dir_shuffled != 0
    X_train = X[mask]
    y_train = y_dir_shuffled[mask]
    
    xgb = XGBoostModel(min_confidence=0.6)
    lgb = LightGBMModel(min_confidence=0.6)
    sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
    rule = RuleBasedModel(min_confidence=0.6)
    
    xgb.train(X_train, y_train)
    lgb.train(X_train, y_train)
    sentiment.train(X_train, y_train)
    rule.train(X_train, y_train)
    
    specialists = [xgb, lgb, sentiment, rule]
    
    # Market features (aligned with training data)
    vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
    news_cols = [col for col in X.columns if 'news' in col.lower()]
    market_features = pd.DataFrame(index=X[mask].index)
    if vol_cols and vol_cols[0] in X_train.columns:
        market_features['volatility'] = X_train[vol_cols[0]]
    else:
        market_features['volatility'] = 0
    if news_cols and news_cols[0] in X_train.columns:
        market_features['news_intensity'] = X_train[news_cols[0]]
    else:
        market_features['news_intensity'] = 0
    market_features = market_features.fillna(0)
    
    # Train ensemble
    ensemble = MetaEnsemble(specialists, min_confidence=0.6)
    ensemble.train(X_train, y_train, market_features)
    
    # Run backtest (using REAL labels for evaluation)
    backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
    test_results = backtest.run(X, prices, y)  # Use real y for evaluation
    
    metrics = test_results['metrics']
    
    result = {
        'test': 'shuffle_labels',
        'accuracy': float(metrics.get('directional_accuracy', 0)),
        'total_return': float(metrics.get('total_return', 0)),
        'sharpe': float(metrics.get('sharpe_ratio', 0)),
        'trades': int((test_results['predictions']['signal'] != 0).sum()),
        'expected_accuracy': 0.5,
        'passed': bool(abs(metrics.get('directional_accuracy', 0) - 0.5) < 0.1 and metrics.get('total_return', 0) < 0.05)
    }
    
    print(f"\nResults with shuffled labels:")
    print(f"  Accuracy: {result['accuracy']:.4f} (expected ~0.5)")
    print(f"  Total Return: {result['total_return']:.4f} (expected ~0)")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  Trades: {result['trades']}")
    print(f"\n{'✓ PASS' if result['passed'] else '✗ FAIL'}: {'No overfitting detected' if result['passed'] else 'Possible overfitting/leakage!'}")
    
    with open(test_results_dir / f"{ticker}_shuffle_test.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def test_time_shift(ticker: str = "AAPL", shifts: List[int] = [-2, -1, 1, 2]) -> pd.DataFrame:
    """
    Test 3: Time-shift test
    Shift features forward/backward by 1-2 days
    Expected: Performance should degrade
    """
    print("\n" + "="*70)
    print("PHASE 1.3: TIME-SHIFT TEST")
    print("="*70)
    
    # Load data
    features = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_prices.csv", index_col=0, parse_dates=True)
    
    feature_cols = [col for col in features.columns 
                    if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
    
    results = []
    
    # Baseline (no shift)
    print("\nBaseline (no shift)...")
    X = features[feature_cols].fillna(0)
    y = features['target_return_1d']
    y_dir = features['target_direction']
    
    mask = y_dir != 0
    X_train = X[mask]
    y_train = y_dir[mask]
    
    xgb = XGBoostModel(min_confidence=0.6)
    lgb = LightGBMModel(min_confidence=0.6)
    sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
    rule = RuleBasedModel(min_confidence=0.6)
    
    xgb.train(X_train, y_train)
    lgb.train(X_train, y_train)
    sentiment.train(X_train, y_train)
    rule.train(X_train, y_train)
    
    specialists = [xgb, lgb, sentiment, rule]
    
    # Market features (aligned with training data)
    vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
    news_cols = [col for col in X.columns if 'news' in col.lower()]
    market_features = pd.DataFrame(index=X[mask].index)
    if vol_cols and vol_cols[0] in X_train.columns:
        market_features['volatility'] = X_train[vol_cols[0]]
    else:
        market_features['volatility'] = 0
    if news_cols and news_cols[0] in X_train.columns:
        market_features['news_intensity'] = X_train[news_cols[0]]
    else:
        market_features['news_intensity'] = 0
    market_features = market_features.fillna(0)
    
    ensemble = MetaEnsemble(specialists, min_confidence=0.6)
    ensemble.train(X_train, y_train, market_features)
    
    backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
    baseline_results = backtest.run(X, prices, y)
    baseline_accuracy = baseline_results['metrics'].get('directional_accuracy', 0)
    
    results.append({
        'shift_days': 0,
        'accuracy': baseline_accuracy,
        'total_return': baseline_results['metrics'].get('total_return', 0),
        'sharpe': baseline_results['metrics'].get('sharpe_ratio', 0)
    })
    
    # Test with shifts
    for shift in shifts:
        print(f"\nTesting shift: {shift} days...")
        
        # Shift features
        X_shifted = X.copy()
        X_shifted = X_shifted.shift(shift)
        X_shifted = X_shifted.fillna(0)
        
        # Train and test
        mask = y_dir != 0
        X_train_shifted = X_shifted[mask]
        y_train = y_dir[mask]
        
        xgb = XGBoostModel(min_confidence=0.6)
        lgb = LightGBMModel(min_confidence=0.6)
        sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
        rule = RuleBasedModel(min_confidence=0.6)
        
        xgb.train(X_train_shifted, y_train)
        lgb.train(X_train_shifted, y_train)
        sentiment.train(X_train_shifted, y_train)
        rule.train(X_train_shifted, y_train)
        
        specialists = [xgb, lgb, sentiment, rule]
        market_features_shifted = market_features.shift(shift).fillna(0)
        ensemble = MetaEnsemble(specialists, min_confidence=0.6)
        ensemble.train(X_train_shifted, y_train, market_features_shifted)
        
        backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
        shift_results = backtest.run(X_shifted, prices, y)
        
        results.append({
            'shift_days': shift,
            'accuracy': shift_results['metrics'].get('directional_accuracy', 0),
            'total_return': shift_results['metrics'].get('total_return', 0),
            'sharpe': shift_results['metrics'].get('sharpe_ratio', 0)
        })
        
        print(f"  Accuracy: {shift_results['metrics'].get('directional_accuracy', 0):.4f} "
              f"(baseline: {baseline_accuracy:.4f})")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(test_results_dir / f"{ticker}_time_shift_test.csv", index=False)
    
    print(f"\n✓ Results saved")
    print(df_results.to_string(index=False))
    
    return df_results


# ============================================================================
# PHASE 2: EXPAND UNIVERSE
# ============================================================================

def test_multi_stock(tickers: List[str] = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"],
                     start_date: str = "2020-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    """
    Test 4: Multi-stock test
    Run pipeline on multiple stocks
    Key metric: Does the pattern hold? (behavior, not just accuracy)
    """
    print("\n" + "="*70)
    print("PHASE 2.1: MULTI-STOCK TEST")
    print("="*70)
    
    results = []
    
    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"Testing {ticker}")
        print('='*70)
        
        try:
            # Fetch data
            price_fetcher = PriceFetcher()
            stock_prices = price_fetcher.fetch(ticker, start_date, end_date)
            index_prices = price_fetcher.fetch_index("^GSPC", start_date, end_date)
            
            news_fetcher = NewsFetcher()
            news_data = news_fetcher.fetch_all(ticker, start_date, end_date)
            
            # Build features
            builder = FeatureBuilder()
            features = builder.build_all_features(stock_prices, index_prices, news_data)
            
            feature_cols = [col for col in features.columns 
                            if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
            X = features[feature_cols].fillna(0)
            y = features['target_return_1d']
            y_dir = features['target_direction']
            
            mask = y_dir != 0
            X_train = X[mask]
            y_train = y_dir[mask]
            
            # Train models
            xgb = XGBoostModel(min_confidence=0.6)
            lgb = LightGBMModel(min_confidence=0.6)
            sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
            rule = RuleBasedModel(min_confidence=0.6)
            
            xgb.train(X_train, y_train)
            lgb.train(X_train, y_train)
            sentiment.train(X_train, y_train)
            rule.train(X_train, y_train)
            
            specialists = [xgb, lgb, sentiment, rule]
            
            # Market features
            vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
            news_cols = [col for col in X.columns if 'news' in col.lower()]
            market_features = pd.DataFrame(index=X_train.index)
            if vol_cols and vol_cols[0] in X_train.columns:
                market_features['volatility'] = X_train[vol_cols[0]]
            else:
                market_features['volatility'] = 0
            if news_cols and news_cols[0] in X_train.columns:
                market_features['news_intensity'] = X_train[news_cols[0]]
            else:
                market_features['news_intensity'] = 0
            market_features = market_features.fillna(0)
            
            ensemble = MetaEnsemble(specialists, min_confidence=0.6)
            ensemble.train(X_train, y_train, market_features)
            
            # Backtest
            backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
            test_results = backtest.run(X, stock_prices, y)
            
            metrics = test_results['metrics']
            predictions = test_results['predictions']
            
            trades = (predictions['signal'] != 0).sum()
            abstentions = (predictions['signal'] == 0).sum()
            coverage = trades / len(predictions) if len(predictions) > 0 else 0
            
            results.append({
                'ticker': ticker,
                'trades': trades,
                'abstentions': abstentions,
                'coverage': coverage,
                'accuracy': float(metrics.get('directional_accuracy', 0)),
                'total_return': float(metrics.get('total_return', 0)),
                'sharpe': float(metrics.get('sharpe_ratio', 0)),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'avg_confidence': predictions['confidence'].mean() if 'confidence' in predictions.columns else 0
            })
            
            print(f"  Trades: {trades}, Coverage: {coverage:.2%}, "
                  f"Accuracy: {metrics.get('directional_accuracy', 0):.4f}, "
                  f"Return: {metrics.get('total_return', 0):.4f}")
            
        except Exception as e:
            print(f"  ✗ Error testing {ticker}: {e}")
            results.append({
                'ticker': ticker,
                'error': str(e)
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(test_results_dir / "multi_stock_test.csv", index=False)
    
    print(f"\n✓ Results saved to: {test_results_dir / 'multi_stock_test.csv'}")
    print("\nSummary:")
    print(df_results.to_string(index=False))
    
    return df_results




def test_cross_stock_generalization(train_tickers: List[str], test_tickers: List[str],
                                     start_date: str = "2020-01-01", end_date: str = "2023-12-31") -> Dict:
    """
    Test 5: Cross-stock generalization
    Train on some stocks, test on unseen stocks
    """
    print("\n" + "="*70)
    print("PHASE 2.2: CROSS-STOCK GENERALIZATION TEST")
    print("="*70)
    print(f"Training on: {train_tickers}")
    print(f"Testing on: {test_tickers}")

    all_X_train = []
    all_y_train = []
    all_market_features_train = []

    for ticker in train_tickers:
        print(f"Loading {ticker} for training...")
        try:
            price_fetcher = PriceFetcher()
            stock_prices = price_fetcher.fetch(ticker, start_date, end_date)
            index_prices = price_fetcher.fetch_index("^GSPC", start_date, end_date)

            news_fetcher = NewsFetcher()
            news_data = news_fetcher.fetch_all(ticker, start_date, end_date)

            builder = FeatureBuilder()
            features = builder.build_all_features(stock_prices, index_prices, news_data)

            feature_cols = [col for col in features.columns
                            if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
            X = features[feature_cols].fillna(0)
            y_dir = features['target_direction']

            mask = y_dir != 0
            all_X_train.append(X[mask])
            all_y_train.append(y_dir[mask])

            vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
            news_cols = [col for col in X.columns if 'news' in col.lower()]
            market_features = pd.DataFrame(index=X[mask].index)
            if vol_cols and vol_cols[0] in X.columns:
                market_features['volatility'] = X.loc[mask, vol_cols[0]]
            else:
                market_features['volatility'] = 0
            if news_cols and news_cols[0] in X.columns:
                market_features['news_intensity'] = X.loc[mask, news_cols[0]]
            else:
                market_features['news_intensity'] = 0
            market_features = market_features.fillna(0)
            all_market_features_train.append(market_features)
        except Exception as e:
            print(f"  ✗ Error loading {ticker}: {e}")

    if not all_X_train:
        print("⚠ No training data loaded")
        return {}

    # Get union of all feature columns to ensure consistency
    all_feature_cols = set()
    for X in all_X_train:
        all_feature_cols.update(X.columns)
    all_feature_cols = sorted(list(all_feature_cols))
    
    # Align all training data to have same columns
    aligned_X_train = []
    for X in all_X_train:
        X_aligned = pd.DataFrame(index=X.index, columns=all_feature_cols)
        for col in all_feature_cols:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                X_aligned[col] = 0
        aligned_X_train.append(X_aligned.fillna(0))
    
    X_train_combined = pd.concat(aligned_X_train, ignore_index=False)
    y_train_combined = pd.concat(all_y_train, ignore_index=False)
    market_features_combined = pd.concat(all_market_features_train, ignore_index=False)

    print(f"\nCombined training data: {len(X_train_combined)} samples")
    print(f"Feature columns: {len(all_feature_cols)}")

    xgb = XGBoostModel(min_confidence=0.6)
    lgb = LightGBMModel(min_confidence=0.6)
    sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
    rule = RuleBasedModel(min_confidence=0.6)

    xgb.train(X_train_combined, y_train_combined)
    lgb.train(X_train_combined, y_train_combined)
    sentiment.train(X_train_combined, y_train_combined)
    rule.train(X_train_combined, y_train_combined)

    specialists = [xgb, lgb, sentiment, rule]
    ensemble = MetaEnsemble(specialists, min_confidence=0.6)
    ensemble.train(X_train_combined, y_train_combined, market_features_combined)

    test_results = {}

    
    for ticker in test_tickers:
        print(f"Testing on unseen stock: {ticker}")
        try:
            price_fetcher = PriceFetcher()
            stock_prices = price_fetcher.fetch(ticker, start_date, end_date)
            index_prices = price_fetcher.fetch_index("^GSPC", start_date, end_date)

            news_fetcher = NewsFetcher()
            news_data = news_fetcher.fetch_all(ticker, start_date, end_date)

            builder = FeatureBuilder()
            features = builder.build_all_features(stock_prices, index_prices, news_data)

            feature_cols = [col for col in features.columns
                            if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
            
            # Align test features to match training features
            X_test = pd.DataFrame(index=features.index, columns=all_feature_cols)
            for col in all_feature_cols:
                if col in feature_cols:
                    X_test[col] = features[col]
                else:
                    X_test[col] = 0
            X_test = X_test.fillna(0)
            y_test = features['target_return_1d']

            vol_cols = [col for col in X_test.columns if 'volatility' in col.lower()]
            news_cols = [col for col in X_test.columns if 'news' in col.lower()]
            market_features_test = pd.DataFrame(index=X_test.index)
            # Always include both market features, even if missing
            if vol_cols and vol_cols[0] in X_test.columns:
                market_features_test['volatility'] = X_test[vol_cols[0]]
            else:
                market_features_test['volatility'] = 0
            if news_cols and news_cols[0] in X_test.columns:
                market_features_test['news_intensity'] = X_test[news_cols[0]]
            else:
                market_features_test['news_intensity'] = 0
            market_features_test = market_features_test.fillna(0)

            predictions = []
            for idx in X_test.index:
                snapshot = X_test.loc[idx]  # Get Series, not DataFrame
                if idx in market_features_test.index:
                    market_snapshot = market_features_test.loc[idx]
                else:
                    market_snapshot = pd.Series({'volatility': 0, 'news_intensity': 0})
                signal = ensemble.get_signal(snapshot, market_snapshot)
                predictions.append(signal)

            predictions_df = pd.DataFrame(predictions, index=X_test.index)
            predictions_df['actual'] = (y_test > 0).astype(int)
            predictions_df['correct'] = (
                ((predictions_df['signal'] == 1) & (predictions_df['actual'] == 1)) |
                ((predictions_df['signal'] == -1) & (predictions_df['actual'] == 0))
            )

            trades = (predictions_df['signal'] != 0).sum()
            accuracy = predictions_df[predictions_df['signal'] != 0]['correct'].mean() if trades > 0 else 0

            test_results[ticker] = {
                'trades': int(trades),
                'accuracy': float(accuracy),
                'coverage': float(trades / len(predictions_df))
            }

            print(f"  Trades: {trades}, Accuracy: {accuracy:.4f}, Coverage: {trades/len(predictions_df):.2%}")
        except Exception as e:
            print(f"  ✗ Error testing {ticker}: {e}")
            test_results[ticker] = {'error': str(e)}

    with open(test_results_dir / "cross_stock_generalization.json", 'w') as f:
        json.dump({
            'train_tickers': train_tickers,
            'test_tickers': test_tickers,
            'results': test_results
        }, f, indent=2)

        print(f"✓ Results saved")

    return test_results


# ============================================================================
# PHASE 3: STRESS ENSEMBLE LOGIC
# ============================================================================

# ============================================================================

def test_disable_components(ticker: str = "AAPL", config: str = None) -> pd.DataFrame:
    """
    Test 6: Disable components one by one
    Run with: no sentiment, no rules, no LGBM, no meta-gating
    Goal: Identify which component actually adds signal
    
    Args:
        ticker: Stock ticker to test
        config: Specific configuration to run (None = all, 'baseline', 'no_sentiment', etc.)
    """
    print("\n" + "="*70)
    print("PHASE 3.1: DISABLE COMPONENTS TEST")
    print("="*70)
    if config:
        print(f"Running configuration: {config}")
    
    # Fetch and build features
    try:
        price_fetcher = PriceFetcher()
        stock_prices = price_fetcher.fetch(ticker, "2020-01-01", "2023-12-31")
        index_prices = price_fetcher.fetch_index("^GSPC", "2020-01-01", "2023-12-31")
        
        news_fetcher = NewsFetcher()
        news_data = news_fetcher.fetch_all(ticker, "2020-01-01", "2023-12-31")
        
        builder = FeatureBuilder()
        features = builder.build_all_features(stock_prices, index_prices, news_data)
        
        feature_cols = [col for col in features.columns 
                        if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
        X = features[feature_cols].fillna(0)
        y = features['target_return_1d']
        y_dir = features['target_direction']
        
        mask = y_dir != 0
        X_train = X[mask]
        y_train = y_dir[mask]
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
    news_cols = [col for col in X.columns if 'news' in col.lower()]
    market_features = pd.DataFrame(index=X[mask].index)
    if vol_cols:
        market_features['volatility'] = X_train[vol_cols[0]] if vol_cols[0] in X_train.columns else X_train.iloc[:, 0] * 0
    if news_cols:
        market_features['news_intensity'] = X_train[news_cols[0]] if news_cols[0] in X_train.columns else X_train.iloc[:, 0] * 0
    market_features = market_features.fillna(0)
    
    # Load existing results if resuming
    results_file = test_results_dir / f"{ticker}_disable_components_test.csv"
    existing_results = {}
    if results_file.exists():
        try:
            existing_df = pd.read_csv(results_file)
            existing_results = {row['configuration']: row for _, row in existing_df.iterrows()}
            print(f"✓ Loaded {len(existing_results)} existing results")
        except:
            pass
    
    results = []
    
    # Simple average ensemble class
    class SimpleAverageEnsemble:
        def __init__(self, specialists, min_confidence=0.6):
            self.specialists = specialists
            self.min_confidence = min_confidence
            self.is_trained = True
        
        def get_signal(self, snapshot, market_features=None):
            signals = []
            confidences = []
            for specialist in self.specialists:
                signal = specialist.get_signal(snapshot)
                signals.append(signal['signal'])
                confidences.append(signal['confidence'])
            
            avg_signal = np.mean(signals)
            avg_confidence = np.mean(confidences)
            
            if avg_confidence < self.min_confidence:
                return {'signal': 0, 'prob_up': 0.5, 'confidence': avg_confidence}
            
            final_signal = 1 if avg_signal > 0.1 else (-1 if avg_signal < -0.1 else 0)
            return {'signal': final_signal, 'prob_up': (avg_signal + 1) / 2, 'confidence': avg_confidence}
        
        def predict_batch(self, X, market_features=None):
            """Predict on a batch of samples using batch operations."""
            # Get batch predictions from all specialists
            specialist_preds = {}
            for specialist in self.specialists:
                if not specialist.is_trained:
                    continue
                try:
                    preds = specialist.predict_batch(X)
                    specialist_preds[specialist.name] = preds
                except Exception as e:
                    print(f"Error from {specialist.name} in predict_batch: {e}")
                    continue
            
            if not specialist_preds:
                # Return abstention for all
                return pd.DataFrame({
                    'signal': 0,
                    'prob_up': 0.5,
                    'confidence': 0.0
                }, index=X.index)
            
            # Average signals and confidences
            signals_list = [preds['signal'].values for preds in specialist_preds.values()]
            confidences_list = [preds['confidence'].values for preds in specialist_preds.values()]
            prob_up_list = [preds['prob_up'].values for preds in specialist_preds.values()]
            
            avg_signal = np.mean(signals_list, axis=0)
            avg_confidence = np.mean(confidences_list, axis=0)
            avg_prob_up = np.mean(prob_up_list, axis=0)
            
            # Apply abstention threshold
            final_signal = np.where(
                avg_confidence < self.min_confidence,
                0,
                np.where(avg_signal > 0.1, 1, np.where(avg_signal < -0.1, -1, 0))
            )
            
            return pd.DataFrame({
                'signal': final_signal,
                'prob_up': avg_prob_up,
                'confidence': avg_confidence
            }, index=X.index)
        
        def train(self, X, y, market_features=None):
            pass  # No training needed for simple average
    
    # Train all specialists once (reused for all configs)
    print("\nTraining all specialists...")
    xgb = XGBoostModel(min_confidence=0.6)
    lgb = LightGBMModel(min_confidence=0.6)
    sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
    rule = RuleBasedModel(min_confidence=0.6)
    
    xgb.train(X_train, y_train)
    lgb.train(X_train, y_train)
    sentiment.train(X_train, y_train)
    rule.train(X_train, y_train)
    
    # Define all configurations
    all_configs = [
        ('baseline', [xgb, lgb, sentiment, rule], False),
        ('no_sentiment', [xgb, lgb, rule], False),
        ('no_rules', [xgb, lgb, sentiment], False),
        ('no_lgbm', [xgb, sentiment, rule], False),
        ('no_xgb', [lgb, sentiment, rule], False),
        ('xgb_only', [xgb], False),
        ('simple_average', [xgb, lgb, sentiment, rule], True),
    ]
    
    # Filter to specific config if requested
    if config:
        all_configs = [c for c in all_configs if c[0] == config]
        if not all_configs:
            print(f"✗ Configuration '{config}' not found")
            return pd.DataFrame()
    
    # Run each configuration
    for config_name, specialists_list, use_simple_avg in all_configs:
        # Skip if already done
        if config_name in existing_results:
            print(f"\n⏭ Skipping {config_name} (already completed)")
            results.append(existing_results[config_name])
            continue
        
        print(f"\nTesting: {config_name}...")
        
        try:
            if use_simple_avg:
                ensemble = SimpleAverageEnsemble(specialists_list, min_confidence=0.6)
            else:
                ensemble = MetaEnsemble(specialists_list, min_confidence=0.6)
                ensemble.train(X_train, y_train, market_features)
            
            backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
            test_results = backtest.run(X, stock_prices, y)
            metrics = test_results['metrics']
            
            result = {
                'configuration': config_name,
                'accuracy': float(metrics.get('directional_accuracy', 0)),
                'total_return': float(metrics.get('total_return', 0)),
                'sharpe': float(metrics.get('sharpe_ratio', 0)),
                'trades': int((test_results['predictions']['signal'] != 0).sum())
            }
            results.append(result)
            
            # Save incrementally
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(results_file, index=False)
            
            print(f"  ✓ Accuracy: {result['accuracy']:.4f}, "
                  f"Return: {result['total_return']:.4f}, "
                  f"Trades: {result['trades']}")
        except Exception as e:
            print(f"  ✗ Error testing {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(test_results_dir / f"{ticker}_disable_components_test.csv", index=False)
    
    print(f"\n✓ Results saved")
    print(df_results.to_string(index=False))
    
    return df_results


# ============================================================================
# PHASE 4: REALITY ALIGNMENT
# ============================================================================

def test_with_transaction_costs(ticker: str = "AAPL", costs_bps: List[float] = [0, 5, 10, 20, 50]) -> pd.DataFrame:
    """
    Test 8: Add transaction costs
    Add 5-10 bps per trade, small slippage noise
    If PnL collapses → edge was fake
    """
    print("\n" + "="*70)
    print("PHASE 4.1: TRANSACTION COSTS TEST")
    print("="*70)
    
    # Load data and run backtest with different commission rates
    try:
        price_fetcher = PriceFetcher()
        stock_prices = price_fetcher.fetch(ticker, "2020-01-01", "2023-12-31")
        index_prices = price_fetcher.fetch_index("^GSPC", "2020-01-01", "2023-12-31")
        
        news_fetcher = NewsFetcher()
        news_data = news_fetcher.fetch_all(ticker, "2020-01-01", "2023-12-31")
        
        builder = FeatureBuilder()
        features = builder.build_all_features(stock_prices, index_prices, news_data)
        
        feature_cols = [col for col in features.columns 
                        if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
        X = features[feature_cols].fillna(0)
        y = features['target_return_1d']
        y_dir = features['target_direction']
        
        mask = y_dir != 0
        X_train = X[mask]
        y_train = y_dir[mask]
        
        # Train models
        xgb = XGBoostModel(min_confidence=0.5)
        lgb = LightGBMModel(min_confidence=0.5)
        sentiment = SentimentModel(min_confidence=0.5, use_pretrained=False)
        rule = RuleBasedModel(min_confidence=0.5)
        
        xgb.train(X_train, y_train)
        lgb.train(X_train, y_train)
        sentiment.train(X_train, y_train)
        rule.train(X_train, y_train)
        
        specialists = [xgb, lgb, sentiment, rule]
        
        vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
        news_cols = [col for col in X.columns if 'news' in col.lower()]
        market_features = pd.DataFrame(index=X_train.index)
        if vol_cols and vol_cols[0] in X_train.columns:
            market_features['volatility'] = X_train[vol_cols[0]]
        else:
            market_features['volatility'] = 0
        if news_cols and news_cols[0] in X_train.columns:
            market_features['news_intensity'] = X_train[news_cols[0]]
        else:
            market_features['news_intensity'] = 0
        market_features = market_features.fillna(0)
        
        results = []
        
        for cost_bps in costs_bps:
            print(f"\nTesting with {cost_bps} bps transaction cost...")
            
            commission_rate = cost_bps / 10000  # Convert bps to decimal
            
            ensemble = MetaEnsemble(specialists, min_confidence=0.5)
            ensemble.train(X_train, y_train, market_features)
            
            backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21, commission_rate=commission_rate)
            test_results = backtest.run(X, stock_prices, y)
            
            metrics = test_results['metrics']
            predictions = test_results['predictions']
            
            trades = (predictions['signal'] != 0).sum()
            
            results.append({
                'cost_bps': cost_bps,
                'trades': int(trades),
                'accuracy': float(metrics.get('directional_accuracy', 0)),
                'total_return': float(metrics.get('total_return', 0)),
                'sharpe': float(metrics.get('sharpe_ratio', 0)),
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'coverage': float(metrics.get('coverage', 0))
            })
            
            print(f"  Trades: {trades}, Accuracy: {metrics.get('directional_accuracy', 0):.4f}, "
                  f"Return: {metrics.get('total_return', 0):.4f}, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(test_results_dir / f"{ticker}_transaction_costs_test.csv", index=False)
        
        print(f"\n✓ Results saved")
        print(df_results.to_string(index=False))
        
        return df_results
        
    except Exception as e:
        print(f"✗ Error in transaction costs test: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def test_minimum_trade_frequency(ticker: str = "AAPL", min_trades_per_period: int = 10) -> pd.DataFrame:
    """
    Test 9: Minimum trade frequency constraint
    Force model to take responsibility by requiring minimum trades
    Accuracy will drop - that's expected and good (reality check)
    """
    print("\n" + "="*70)
    print("PHASE 4.2: MINIMUM TRADE FREQUENCY TEST")
    print("="*70)
    
    try:
        price_fetcher = PriceFetcher()
        stock_prices = price_fetcher.fetch(ticker, "2020-01-01", "2023-12-31")
        index_prices = price_fetcher.fetch_index("^GSPC", "2020-01-01", "2023-12-31")
        
        news_fetcher = NewsFetcher()
        news_data = news_fetcher.fetch_all(ticker, "2020-01-01", "2023-12-31")
        
        builder = FeatureBuilder()
        features = builder.build_all_features(stock_prices, index_prices, news_data)
        
        feature_cols = [col for col in features.columns 
                        if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
        X = features[feature_cols].fillna(0)
        y = features['target_return_1d']
        y_dir = features['target_direction']
        
        mask = y_dir != 0
        X_train = X[mask]
        y_train = y_dir[mask]
        
        # Test different confidence thresholds to force more trades
        thresholds = [0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
        results = []
        
        for threshold in thresholds:
            print(f"\nTesting with confidence threshold: {threshold}")
            
            xgb = XGBoostModel(min_confidence=threshold)
            lgb = LightGBMModel(min_confidence=threshold)
            sentiment = SentimentModel(min_confidence=threshold, use_pretrained=False)
            rule = RuleBasedModel(min_confidence=threshold)
            
            xgb.train(X_train, y_train)
            lgb.train(X_train, y_train)
            sentiment.train(X_train, y_train)
            rule.train(X_train, y_train)
            
            specialists = [xgb, lgb, sentiment, rule]
            
            vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
            news_cols = [col for col in X.columns if 'news' in col.lower()]
            market_features = pd.DataFrame(index=X_train.index)
            if vol_cols and vol_cols[0] in X_train.columns:
                market_features['volatility'] = X_train[vol_cols[0]]
            else:
                market_features['volatility'] = 0
            if news_cols and news_cols[0] in X_train.columns:
                market_features['news_intensity'] = X_train[news_cols[0]]
            else:
                market_features['news_intensity'] = 0
            market_features = market_features.fillna(0)
            
            ensemble = MetaEnsemble(specialists, min_confidence=threshold)
            ensemble.train(X_train, y_train, market_features)
            
            backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
            test_results = backtest.run(X, stock_prices, y)
            
            metrics = test_results['metrics']
            predictions = test_results['predictions']
            
            trades = (predictions['signal'] != 0).sum()
            
            results.append({
                'threshold': threshold,
                'trades': int(trades),
                'accuracy': float(metrics.get('directional_accuracy', 0)),
                'total_return': float(metrics.get('total_return', 0)),
                'sharpe': float(metrics.get('sharpe_ratio', 0)),
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'coverage': float(metrics.get('coverage', 0)),
                'meets_minimum': trades >= min_trades_per_period
            })
            
            print(f"  Trades: {trades}, Accuracy: {metrics.get('directional_accuracy', 0):.4f}, "
                  f"Return: {metrics.get('total_return', 0):.4f}")
            
            if trades >= min_trades_per_period:
                print(f"  ✓ Meets minimum trade requirement ({min_trades_per_period})")
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(test_results_dir / f"{ticker}_minimum_trade_frequency_test.csv", index=False)
        
        print(f"\n✓ Results saved")
        print(df_results.to_string(index=False))
        
        # Find threshold that meets minimum
        valid_configs = df_results[df_results['meets_minimum'] == True]
        if len(valid_configs) > 0:
            best = valid_configs.loc[valid_configs['accuracy'].idxmax()]
            print(f"\n✓ Best configuration meeting minimum ({min_trades_per_period} trades):")
            print(f"  Threshold: {best['threshold']}, Trades: {best['trades']}, Accuracy: {best['accuracy']:.4f}")
        else:
            print(f"\n⚠ No configuration meets minimum trade requirement ({min_trades_per_period})")
        
        return df_results
        
    except Exception as e:
        print(f"✗ Error in minimum trade frequency test: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run robustness tests')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5], help='Phase to run (1-5)')
    parser.add_argument('--all', action='store_true', help='Run all phases')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker to test')
    parser.add_argument('--config', type=str, help='For Phase 3: specific config to run (baseline, no_sentiment, no_rules, no_lgbm, no_xgb, xgb_only, simple_average)')
    
    args = parser.parse_args()
    
    if args.all or args.phase == 1:
        print("\n" + "="*70)
        print("RUNNING PHASE 1: SANITY & ROBUSTNESS CHECKS")
        print("="*70)
        test_confidence_thresholds(args.ticker)
        test_shuffle_labels(args.ticker)
        test_time_shift(args.ticker)
    
    if args.all or args.phase == 2:
        print("\n" + "="*70)
        print("RUNNING PHASE 2: EXPAND UNIVERSE")
        print("="*70)
        test_multi_stock()
        test_cross_stock_generalization(["AAPL", "MSFT", "NVDA", "AMZN"], ["META", "TSLA"])
    
    if args.all or args.phase == 3:
        print("\n" + "="*70)
        print("RUNNING PHASE 3: STRESS ENSEMBLE LOGIC")
        print("="*70)
        test_disable_components(args.ticker, config=args.config)
    
    if args.all or args.phase == 4:
        print("\n" + "="*70)
        print("RUNNING PHASE 4: REALITY ALIGNMENT")
        print("="*70)
        test_with_transaction_costs(args.ticker)
        test_minimum_trade_frequency(args.ticker)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {test_results_dir}")
    print("\nReview results and decide if the model has real edge!")

