#!/usr/bin/env python3
"""
Adversarial Validation - Phase E
Tests robustness of the Structural Alpha.
E1. Random Regime Test (Shuffle Labels)
E2. Cross-Asset Generalization (Train AAPL -> Test MSFT/GOOGL)
"""

import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime
from strategies.strategy_wrapper import StrategyWrapper

def get_data_and_features(ticker):
    print(f"Fetching data for {ticker}...")
    pf = PriceFetcher()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=3650)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    prices = pf.fetch(ticker, start_date=start_date, end_date=end_date)
    
    print("Building features...")
    fb = FeatureBuilder()
    features = fb.build_all_features(prices)
    
    print(f"Labeling history (Ground Truth) for {ticker}...")
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(prices)
    
    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx, 'regime']
    
    # Binarize
    y_binary = y.apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    
    # Clean X
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume', 'regime', 'regime_name']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    return prices, X, y_binary

def train_model(X_train, y_train):
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_strategy(prices, predicted_regime_binary):
    # Map: 0 (Risk On) -> Momentum (TRENDING), 1 (Risk Off) -> Flat (HIGH_VOL)
    mapped_regime = predicted_regime_binary.map({0: MarketRegime.TRENDING, 1: MarketRegime.HIGH_VOL})
    
    strat = StrategyWrapper()
    # Align prices
    test_prices = prices.loc[mapped_regime.index]
    signals = strat.generate_signals(test_prices, mapped_regime)
    # Cost 10bps
    results = strat.backtest_conditional(test_prices, signals, cost_bps=10.0)
    return results['total_return'], results['sharpe']

def run_adversarial_suite():
    print("=== Phase E: Adversarial Validation ===")
    
    # Get AAPL Data (Source)
    prices_aapl, X_aapl, y_aapl = get_data_and_features('AAPL')
    
    # Train Model on AAPL (Full History for simplicity of Transfer Test, or Split?)
    # For E2, we Train on AAPL, Test on MSFT.
    print("\n--- E2. Cross-Asset Generalization (Train AAPL -> Predict MSFT) ---")
    
    # Train on AAPL (First 70% or All? Let's use All to maximize signal learning)
    print("Training Regime Detector on AAPL...")
    model_aapl = train_model(X_aapl, y_aapl)
    
    # Evaluate on MSFT
    print("Testing on MSFT...")
    prices_msft, X_msft, y_msft_true = get_data_and_features('MSFT')
    
    # Predict MSFT Regimes using AAPL Model
    preds_msft = model_aapl.predict(X_msft)
    pred_series_msft = pd.Series(preds_msft, index=X_msft.index)
    
    # Evaluate Strategy on MSFT
    ret_msft, sharpe_msft = evaluate_strategy(prices_msft, pred_series_msft)
    
    # Benchmark MSFT (Unconditional Momentum)
    mapped_uncond = pd.Series(MarketRegime.TRENDING, index=pred_series_msft.index)
    strat = StrategyWrapper()
    test_prices_msft = prices_msft.loc[mapped_uncond.index]
    sig_uncond = strat.generate_signals(test_prices_msft, mapped_uncond)
    res_uncond = strat.backtest_conditional(test_prices_msft, sig_uncond, cost_bps=10.0)
    
    delta_msft = ret_msft - res_uncond['total_return']
    print(f"MSFT Conditional Return: {ret_msft:.2%}")
    print(f"MSFT Unconditional Mom:  {res_uncond['total_return']:.2%}")
    print(f"MSFT Delta:              {delta_msft:.2%}")
    
    if delta_msft > 0:
        print("PASS: Structural Alpha generalizes to MSFT.")
    else:
        print("FAIL: Alpha does not transfer.")
        
    # --- E1: Random Regime Test (Shuffle) ---
    print("\n--- E1. Random Regime Test (AAPL) ---")
    # We take the TRUE predictions on AAPL (In-Sample/Walk-Forward) and shuffle them.
    # Use Walk-Forward predictions from previous step logic?
    # Let's verify on the MSFT predictions (Out of Sample).
    # We have `pred_series_msft`.
    # Shuffle this series 100 times and check PnL.
    
    actual_ret = ret_msft
    random_returns = []
    
    print("Running 100 Shuffles...")
    for i in range(100):
        shuffled_preds = pred_series_msft.sample(frac=1.0, replace=False).values
        shuffled_series = pd.Series(shuffled_preds, index=pred_series_msft.index)
        
        r, s = evaluate_strategy(prices_msft, shuffled_series)
        random_returns.append(r)
    
    avg_random = np.mean(random_returns)
    pct_95 = np.percentile(random_returns, 95)
    
    print(f"Actual Return:      {actual_ret:.2%}")
    print(f"Avg Random Return:  {avg_random:.2%}")
    print(f"95th Pct Random:    {pct_95:.2%}")
    
    if actual_ret > pct_95:
        print("PASS: Performance is significantly better than random (p < 0.05).")
    else:
        print("FAIL: Performance is indistinguishable from random noise.")

if __name__ == "__main__":
    run_adversarial_suite()
