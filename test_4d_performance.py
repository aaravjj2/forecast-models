#!/usr/bin/env python3
"""
Test 4D Lattice Performance - Phase D2
Walk-forward backtest of the 4D Regime Lattice Strategy.
Filters: Volatility, Trend Quality, Liquidity, Info State.
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality, LiquidityRegime, InfoRegime
from strategies.regime_lattice import RegimeLattice
from strategies.strategy_wrapper import StrategyWrapper

def run_backtest():
    print("=== D2. 4D Lattice Performance Test (AAPL) ===")
    
    # 1. Data
    ticker = 'AAPL'
    print(f"Loading cached data for {ticker}")
    pf = PriceFetcher()
    prices = pf.fetch(ticker, start_date='2015-01-01', end_date='2024-01-01')
    
    # 2. Features
    print("Building features...")
    fb = FeatureBuilder()
    features = fb.build_all_features(prices)
    
    # 3. Targets
    print("Labeling Ground Truths...")
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(prices)
    
    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y_labels = labels.loc[common_idx]
    
    # Drop lookahead
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # Binary Targets
    y_vol = y_labels['regime'].apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    y_trend = y_labels['trend_quality'].apply(lambda x: 1 if x == TrendQuality.ROBUST else 0)
    y_liq = y_labels['liquidity_regime'].apply(lambda x: 1 if x == LiquidityRegime.STRESSED else 0)
    y_info = y_labels['info_regime'].apply(lambda x: 1 if x == InfoRegime.DRIFTING else 0)
    
    # 4. Walk-Forward Prediction
    print("Running 4D Walk-Forward Prediction...")
    
    # Expanding Window: Initial Train 2 years (500 days), then retrain every year (250 days)?
    # Or simple train/test for speed in this test script?
    # User requirement: "Walk-forward validation".
    # Let's do a simple rolling simulate: Train on past, predict next window.
    
    train_size = 500
    step_size = 20 # Monthly retrain
    
    preds_vol = []
    preds_trend = []
    preds_liq = []
    preds_info = []
    indices = []
    
    # Use a faster approach: Train on 0..T, Predict T+1..T+Step
    # Start loop
    for t in range(train_size, len(X), step_size):
        end_idx = min(t + step_size, len(X))
        
        X_train = X.iloc[:t]
        X_test = X.iloc[t:end_idx]
        
        # Train 4 Models
        # Vol
        clf_vol = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
        clf_vol.fit(X_train, y_vol.iloc[:t])
        p_vol = clf_vol.predict(X_test)
        
        # Trend
        clf_trend = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
        clf_trend.fit(X_train, y_trend.iloc[:t])
        p_trend = clf_trend.predict(X_test)
        
        # Liq
        clf_liq = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
        clf_liq.fit(X_train, y_liq.iloc[:t])
        p_liq = clf_liq.predict(X_test)
        
        # Info
        clf_info = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1, n_jobs=1)
        clf_info.fit(X_train, y_info.iloc[:t])
        p_info = clf_info.predict(X_test)
        
        preds_vol.extend(p_vol)
        preds_trend.extend(p_trend)
        preds_liq.extend(p_liq)
        preds_info.extend(p_info)
        indices.extend(X_test.index)
        
        if t % 500 == 0:
            print(f"  Step {t}/{len(X)}...")

    # Assemble Result
    res_df = pd.DataFrame({
        'vol': preds_vol,
        'trend': preds_trend,
        'liq': preds_liq,
        'info': preds_info
    }, index=indices)
    
    # 5. Lattice Logic
    lattice = RegimeLattice()
    states = lattice.determine_state(res_df['vol'], res_df['trend'], res_df['liq'], res_df['info'])
    
    # 6. Strategy Execution
    print("Executing 4D Strategy...")
    strat = StrategyWrapper()
    signals = strat.generate_signals(prices.loc[states.index], states)
    result = strat.backtest_conditional(prices.loc[states.index], signals)
    
    # 7. Baseline (Vol Gating Only)
    # Vol=1 -> Hostile. Vol=0 -> Neutral/Favorable (Momentum).
    # If Vol=0, use Trend? No, baseline is simple Vol Gating from Phase 3.
    # Phase 3: Vol=1 -> Flat. Vol=0 -> Momentum.
    baseline_signals = pd.Series(0.0, index=states.index)
    # Recreate Phase 3 logic:
    # Calculate Momentum
    ma_50 = prices['close'].rolling(50).mean().loc[states.index]
    mom = np.where(prices.loc[states.index, 'close'] > ma_50, 1.0, -1.0)
    # Filter by Vol
    mask_safe = (res_df['vol'] == 0)
    baseline_signals[mask_safe] = mom[mask_safe]
    
    baseline_res = strat.backtest_conditional(prices.loc[states.index], baseline_signals)
    
    print("\n=== 4D LATTICE vs BASELINE (AAPL) ===")
    
    print("\n--- Baseline (Vol Gating Only) ---")
    print(f"Return:   {baseline_res['total_return']:.2%}")
    print(f"Sharpe:   {baseline_res['sharpe']:.2f}")
    print(f"Trades:   {baseline_res['n_trades']}")
    print(f"Turnover: {baseline_res['turnover_count']:.2f}")

    print("\n--- 4D Stacked (Vol + Trend + Liq + Info) ---")
    print(f"Return:   {result['total_return']:.2%}")
    print(f"Sharpe:   {result['sharpe']:.2f}")
    print(f"Trades:   {result['n_trades']}")
    print(f"Turnover: {result['turnover_count']:.2f}")
    
    delta_ret = result['total_return'] - baseline_res['total_return']
    delta_sharpe = result['sharpe'] - baseline_res['sharpe']
    
    print(f"\nDelta Return: {delta_ret:.2%}")
    print(f"Delta Sharpe: {delta_sharpe:.2f}")
    
    if delta_ret > 0 and delta_sharpe > 0:
        print("PASS: 4D Stacking adds Value.")
    else:
        print("FAIL: Stacking is noise/duplicates.")

if __name__ == "__main__":
    run_backtest()
