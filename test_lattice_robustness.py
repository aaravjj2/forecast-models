#!/usr/bin/env python3
"""
Test Lattice Robustness - Phase E
Stress tests the Stacked Strategy (Regime Lattice).
4D Version (Vol, Trend, Liq, Info).
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
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality, LiquidityRegime, InfoRegime
from strategies.strategy_wrapper import StrategyWrapper
from strategies.regime_lattice import RegimeLattice, LatticeState

def get_data_features_targets(ticker):
    print(f"Fetching {ticker}...")
    pf = PriceFetcher()
    prices = pf.fetch(ticker, start_date='2015-01-01', end_date=pd.Timestamp.now().strftime('%Y-%m-%d'))
    fb = FeatureBuilder()
    features = fb.build_all_features(prices)
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(prices)
    
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    
    # Drop lookahead
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # Targets
    y_vol = labels.loc[common_idx, 'regime'].apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    y_trend = labels.loc[common_idx, 'trend_quality'].apply(lambda x: 1 if x == TrendQuality.ROBUST else 0)
    y_liq = labels.loc[common_idx, 'liquidity_regime'].apply(lambda x: 1 if x == LiquidityRegime.STRESSED else 0)
    y_info = labels.loc[common_idx, 'info_regime'].apply(lambda x: 1 if x == InfoRegime.DRIFTING else 0)
    
    return prices, X, y_vol, y_trend, y_liq, y_info

def train_models(X, y_vol, y_trend, y_liq, y_info):
    print("Training 4 Models...")
    params = {'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42, 'verbose': -1}
    
    clf_vol = lgb.LGBMClassifier(**params).fit(X, y_vol)
    clf_trend = lgb.LGBMClassifier(**params).fit(X, y_trend)
    clf_liq = lgb.LGBMClassifier(**params).fit(X, y_liq)
    clf_info = lgb.LGBMClassifier(**params).fit(X, y_info)
    
    return clf_vol, clf_trend, clf_liq, clf_info

def run_lattice_strategy(prices, lattice_states, prediction_df, cost_bps=10.0):
    strat = StrategyWrapper()
    # Baseline logic inside wrapper needs raw data, but if we pass predicted_regime=LatticeState, it uses 4D logic.
    # To run Baseline (Vol Only), we need to manually pass Vol=0/1 mapped to Hostile/Favorable?
    # No, StrategyWrapper has baseline logic built-in? No, I implemented hardcoded Lattice logic.
    # So I need to generate baseline signals externally or trick the wrapper.
    # Let's trust the Wrapper to execute the Lattice States given.
    
    test_prices = prices.loc[lattice_states.index]
    signals = strat.generate_signals(test_prices, lattice_states)
    res = strat.backtest_conditional(test_prices, signals, cost_bps=cost_bps)
    return res

def run_baseline(prices, states_index, vol_preds, cost_bps=10.0):
    # Baseline: Vol Gating Only.
    # Vol=0 -> Momentum (1.0). Vol=1 -> Flat.
    test_prices = prices.loc[states_index]
    signals = pd.Series(0.0, index=states_index)
    
    ma_50 = test_prices['close'].rolling(50).mean()
    mom = np.where(test_prices['close'] > ma_50, 1.0, -1.0)
    
    mask_safe = (vol_preds == 0)
    signals[mask_safe] = mom[mask_safe]
    
    strat = StrategyWrapper()
    res = strat.backtest_conditional(test_prices, signals, cost_bps=cost_bps)
    return res

def run_robustness_suite():
    print("=== Phase E: Lattice Robustness Test (4D) ===")
    
    # Get AAPL Data (Source)
    prices_aapl, X_aapl, y_vol_aapl, y_trend_aapl, y_liq_aapl, y_info_aapl = get_data_features_targets('AAPL')
    
    # Train In-Sample (Transfer Source)
    m_vol, m_trend, m_liq, m_info = train_models(X_aapl, y_vol_aapl, y_trend_aapl, y_liq_aapl, y_info_aapl)
    
    # ----------------------------------------------------------------
    # E2. Cross-Asset Transfer (MSFT)
    # ----------------------------------------------------------------
    print("\n--- E2. Cross-Asset Transfer (Train AAPL -> Predict MSFT) ---")
    prices_msft, X_msft, _, _, _, _ = get_data_features_targets('MSFT')
    
    # Predict
    p_vol = m_vol.predict(X_msft)
    p_trend = m_trend.predict(X_msft)
    p_liq = m_liq.predict(X_msft)
    p_info = m_info.predict(X_msft)
    
    pred_df = pd.DataFrame({'vol': p_vol, 'trend': p_trend, 'liq': p_liq, 'info': p_info}, index=X_msft.index)
    
    # Lattice Construction
    lattice = RegimeLattice()
    states_msft = lattice.determine_state(pred_df['vol'], pred_df['trend'], pred_df['liq'], pred_df['info'])
    
    # Backtest Stacked
    res_msft = run_lattice_strategy(prices_msft, states_msft, pred_df, cost_bps=10.0)
    print(f"MSFT 4D Return: {res_msft['total_return']:.2%}")
    print(f"MSFT Sharpe:    {res_msft['sharpe']:.2f}")
    
    # Benchmark MSFT (Vol Only)
    res_msft_bm = run_baseline(prices_msft, states_msft.index, pred_df['vol'], cost_bps=10.0)
    print(f"MSFT Baseline:  {res_msft_bm['total_return']:.2%}")
    print(f"MSFT BM Sharpe: {res_msft_bm['sharpe']:.2f}")
    
    delta_ret = res_msft['total_return'] - res_msft_bm['total_return']
    delta_sharpe = res_msft['sharpe'] - res_msft_bm['sharpe']
    print(f"MSFT Delta Ret: {delta_ret:.2%}")
    print(f"MSFT Delta Shp: {delta_sharpe:.2f}")
    
    if delta_ret > 0:
        print("PASS: 4D Edge generalizes to MSFT.")
    else:
        print("FAIL: Stacked Edge does not transfer.")
        
    # ----------------------------------------------------------------
    # E1. Regime Shuffle (on MSFT)
    # ----------------------------------------------------------------
    print("\n--- E1. Regime Shuffle Test (MSFT) ---")
    actual_ret = res_msft['total_return']
    random_rets = []
    
    for _ in range(50):
        # Shuffle the Lattice States
        shuffled = states_msft.sample(frac=1.0, replace=False).values
        shuffled_states = pd.Series(shuffled, index=states_msft.index)
        # Using prediction_df just for signature, not used inside for shuffled
        res = run_lattice_strategy(prices_msft, shuffled_states, pred_df, cost_bps=10.0)
        random_rets.append(res['total_return'])
        
    avg_rnd = np.mean(random_rets)
    pct95_rnd = np.percentile(random_rets, 95)
    
    print(f"Actual Return:     {actual_ret:.2%}")
    print(f"Avg Random Return: {avg_rnd:.2%}")
    print(f"95th Pct Random:   {pct95_rnd:.2%}")
    
    if actual_ret > pct95_rnd:
        print("PASS: Lattice structure is significant (p < 0.05).")
    else:
        print("FAIL: Lattice is indistinguishable from random noise.")

if __name__ == "__main__":
    run_robustness_suite()
