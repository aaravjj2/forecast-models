#!/usr/bin/env python3
"""
Test Stacked Performance - Phase D2
Backtests the Multi-Regime Lattice Strategy (Volatility + Trend Quality)
vs Single Regime Baselines.
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
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality
from strategies.strategy_wrapper import StrategyWrapper
from strategies.regime_lattice import RegimeLattice, LatticeState

def run_stacked_test(ticker='AAPL'):
    print(f"=== D2. Stacked Performance Test ({ticker}) ===")
    
    # 1. Data & Labeling
    pf = PriceFetcher()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=3650)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    prices = pf.fetch(ticker, start_date=start_date, end_date=end_date)
    
    print("Building features...")
    fb = FeatureBuilder()
    features = fb.build_all_features(prices)
    
    print("Labeling Ground Truths...")
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(prices)
    
    # Features & Targets
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    
    # Target 1: Volatility (Risk Off)
    y_vol = labels.loc[common_idx, 'regime'].apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    
    # Target 2: Trend Quality (Robust)
    y_trend = labels.loc[common_idx, 'trend_quality'].apply(lambda x: 1 if x == TrendQuality.ROBUST else 0)
    
    # Clean X
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume', 'regime', 'regime_name', 'trend_quality', 'trend_quality_name', 'future_r2', 'future_vol_5d', 'future_return_5d']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # 2. Walk-Forward Prediction (Stacked)
    print("Running Stacked Walk-Forward Prediction...")
    min_train = 500
    step = 120
    
    vol_preds = []
    trend_preds = []
    indices = []
    
    for i in range(min_train, len(X), step):
        train_idx = X.index[:i]
        test_idx = X.index[i:i+step]
        
        if len(test_idx) == 0:
            break
            
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        
        # Train Volatility
        y_vol_train = y_vol.loc[train_idx]
        clf_vol = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        clf_vol.fit(X_train, y_vol_train)
        p_vol = clf_vol.predict(X_test)
        
        # Train Trend Quality
        y_trend_train = y_trend.loc[train_idx]
        clf_trend = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        clf_trend.fit(X_train, y_trend_train)
        p_trend = clf_trend.predict(X_test)
        
        vol_preds.extend(p_vol)
        trend_preds.extend(p_trend)
        indices.extend(test_idx)
    
    # 3. Create Lattice States
    pred_vol = pd.Series(vol_preds, index=indices)
    pred_trend = pd.Series(trend_preds, index=indices)
    
    lattice = RegimeLattice()
    lattice_states = lattice.determine_state(pred_vol, pred_trend)
    lattice_states.name = 'lattice_state'
    
    # 4. Backtest Strategy
    print("Executing Stacked Strategies...")
    strat = StrategyWrapper()
    test_prices = prices.loc[lattice_states.index]
    
    # Run Pattern: Stacked (Lattice)
    signals_stacked = strat.generate_signals(test_prices, lattice_states)
    res_stacked = strat.backtest_conditional(test_prices, signals_stacked, cost_bps=10.0)
    
    # Run Pattern: Single Regime (Vol Only) - Baseline from Phase 3
    # Map Vol Prediction to Lattice Logic (ignoring Trend Quality):
    # If Vol=1 -> Hostile. If Vol=0 -> Favorable (Assume Trend is always good for baseline).
    # Baseline: Risk Off -> Flat, Risk On -> Momentum (Full Size)
    # So we map Vol=0 to LatticeState.FAVORABLE, Vol=1 to HOSTILE.
    # We skip NEUTRAL state.
    baseline_states = pred_vol.map({0: LatticeState.FAVORABLE, 1: LatticeState.HOSTILE})
    signals_baseline = strat.generate_signals(test_prices, baseline_states)
    res_baseline = strat.backtest_conditional(test_prices, signals_baseline, cost_bps=10.0)
    
    # Report
    print("\n=== STACKED vs BASELINE RESULTS (AAPL) ===")
    
    print("\n--- Baseline (Volatility Gating Only) ---")
    print(f"Return:   {res_baseline['total_return']:.2%}")
    print(f"Sharpe:   {res_baseline['sharpe']:.2f}")
    print(f"MaxDD:    N/A") # Not calc in simple backtester
    print(f"Trades:   {res_baseline['n_trades']}")
    print(f"Turnover: {res_baseline['turnover_count']:.2f}")

    print("\n--- Stacked (Vol + Trend Quality Lattice) ---")
    print(f"Return:   {res_stacked['total_return']:.2%}")
    print(f"Sharpe:   {res_stacked['sharpe']:.2f}")
    print(f"Trades:   {res_stacked['n_trades']}")
    print(f"Turnover: {res_stacked['turnover_count']:.2f}")
    
    delta_ret = res_stacked['total_return'] - res_baseline['total_return']
    delta_sharpe = res_stacked['sharpe'] - res_baseline['sharpe']
    
    print(f"\nDelta Return: {delta_ret:.2%}")
    print(f"Delta Sharpe: {delta_sharpe:.2f}")
    
    if delta_sharpe > 0.05 or delta_ret > 0.05:
        print("PASS: Stacking adds Value.")
    else:
        print("FAIL: Stacking is noise/duplicates.")

if __name__ == "__main__":
    run_stacked_test()
