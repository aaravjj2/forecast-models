#!/usr/bin/env python3
"""
Test Conditional Performance - Phase D2
Runs the full pipeline:
1. Feature Engineering (Phase B)
2. Regime Labeling (Phase A)
3. Train Regime Detector (Phase C) (Walk-Forward) - Simpler: Train/Test Split
4. Apply Conditional Strategy (Phase D1)
5. Compare Conditional vs Unconditional PnL.
"""

import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime
from strategies.strategy_wrapper import StrategyWrapper

def run_conditional_test(ticker='AAPL'):
    print("=== D2. Conditional Performance Test ===")
    
    # 1. Fetch Data
    pf = PriceFetcher()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=3650)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    prices = pf.fetch(ticker, start_date=start_date, end_date=end_date)
    
    # 2. Features
    print("Building features...")
    fb = FeatureBuilder()
    features = fb.build_all_features(prices)
    
    # 3. Ground Truth Labels (for training)
    print("Labeling history...")
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(prices)
    
    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx, 'regime']
    
    X = features.loc[common_idx]
    y = labels.loc[common_idx, 'regime']
    
    # Binarize Target
    # 0 = Risk On (Trade), 1 = Risk Off (Flat)
    y_binary = y.apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    
    # Drop targets/lookahead from X
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume', 'regime', 'regime_name']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # 4. Walk-Forward Prediction (Simulated Real-Time) - Binary
    print("Running Walk-Forward Regime Prediction (Binary Risk-Off)...")
    
    # Initial training window: 2 years (500 days)
    min_train = 500
    predictions = []
    indices = []
    step = 120
    
    for i in range(min_train, len(X), step):
        train_idx = X.index[:i]
        test_idx = X.index[i:i+step]
        
        if len(test_idx) == 0:
            break
            
        X_train, y_train = X.loc[train_idx], y_binary.loc[train_idx] # Binary Target
        X_test = X.loc[test_idx]
        
        # Train Binary Classifier
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        
        # Predict
        preds = clf.predict(X_test)
        
        predictions.extend(preds)
        indices.extend(test_idx)
        
    pred_series = pd.Series(predictions, index=indices)
    
    # Map Binary Predictions to Strategy Regimes
    # 0 (Risk On) -> MarketRegime.TRENDING (1) -> Triggers Momentum Strategy
    # 1 (Risk Off) -> MarketRegime.HIGH_VOL (2) -> Triggers Flat Strategy
    mapped_predictions = pred_series.map({0: MarketRegime.TRENDING, 1: MarketRegime.HIGH_VOL})
    mapped_predictions.name = 'predicted_regime'
    
    # 5. Execute Strategy
    print("Executing Conditional Strategies...")
    strat = StrategyWrapper()
    
    # Subset prices to prediction period
    test_prices = prices.loc[mapped_predictions.index]
    
    signals = strat.generate_signals(test_prices, mapped_predictions)
    
    results = strat.backtest_conditional(test_prices, signals)
    
    print("\n--- Conditional Strategy Results (Risk On=Mom, Risk Off=Flat) ---")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"Win Rate:     {results['win_rate']:.2%}")
    print(f"Days Active:  {results['n_trades']}")
    print(f"Turnover:     {results['turnover_count']:.2f} units")
    
    # Compare with Unconditional Benchmarks
    # Benchmark 1: Buy & Hold
    bh_ret = (test_prices['close'].iloc[-1] / test_prices['close'].iloc[0]) - 1
    print(f"\n--- Benchmark: Buy & Hold ---")
    print(f"Total Return: {bh_ret:.2%}")
    
    # Benchmark 2: Unconditional Momentum
    # Use TRENDING regime for all
    always_trend = pd.Series(MarketRegime.TRENDING, index=mapped_predictions.index)
    sig_mom = strat.generate_signals(test_prices, always_trend)
    res_mom = strat.backtest_conditional(test_prices, sig_mom)
    print(f"\n--- Benchmark: Unconditional Momentum ---")
    print(f"Total Return: {res_mom['total_return']:.2%}")
    print(f"Sharpe Ratio: {res_mom['sharpe']:.2f}")
    print(f"Turnover:     {res_mom['turnover_count']:.2f} units")

    # Benchmark 3: Unconditional Mean Reversion
    # Use LOW_VOL regime for all (StrategyWrapper maps LowVol/Neutral -> MeanRev)
    always_mr = pd.Series(MarketRegime.LOW_VOL, index=mapped_predictions.index)
    sig_mr = strat.generate_signals(test_prices, always_mr)
    res_mr = strat.backtest_conditional(test_prices, sig_mr)
    print(f"\n--- Benchmark: Unconditional Mean Reversion ---")
    print(f"Total Return: {res_mr['total_return']:.2%}")
    print(f"Sharpe Ratio: {res_mr['sharpe']:.2f}")
    
    # THE DELTA
    delta_mom = results['total_return'] - res_mom['total_return']
    print(f"\n=== EDGE VERIFICATION ===")
    print(f"Conditional vs Uncond Momentum Delta: {delta_mom:.2%}")
    
    if delta_mom > 0:
        print("PASS: Conditional Logic adds value.")
    else:
        print("FAIL: Strategy performs worse than naive approach.")

if __name__ == "__main__":
    run_conditional_test()
