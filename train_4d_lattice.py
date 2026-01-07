#!/usr/bin/env python3
"""
Train 4D Regime Lattice - Phase C
Orchestrates the training of 4 orthogonal regime classifiers:
1. Volatility (Risk Off)
2. Trend Quality (Trend Robustness)
3. Liquidity (Execution Stress)
4. Info State (Momentum Drift)
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, accuracy_score
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality, LiquidityRegime, InfoRegime

def train_and_evaluate():
    print("=== Training 4D Regime Lattice (AAPL) ===")
    
    # 1. Data & Features
    pf = PriceFetcher()
    prices = pf.fetch("AAPL", start_date="2015-01-01", end_date="2024-01-01")
    
    fb = FeatureBuilder()
    # Note: news is None for now unless we implement fetching news. 
    # InfoRegime currently relies on Price Momentum Lag, which is in Price Features.
    # Ideally we'd have news, but "Price Underreaction" proxy works on price/volume too.
    features = fb.build_all_features(prices)
    
    # 2. Ground Truth Labels
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(prices)
    
    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y_labels = labels.loc[common_idx]
    
    # Drop lookahead columns from X
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 
                 'open', 'high', 'low', 'close', 'volume']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # 3. Define Targets
    # A. Volatility: 1 = Risk Off (High Vol or Crash)
    y_vol = y_labels['regime'].apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    
    # B. Trend Quality: 1 = Robust (High ADX)
    y_trend = y_labels['trend_quality'].apply(lambda x: 1 if x == TrendQuality.ROBUST else 0)
    
    # C. Liquidity: 1 = Stressed (High Illiquidity)
    y_liq = y_labels['liquidity_regime'].apply(lambda x: 1 if x == LiquidityRegime.STRESSED else 0)
    
    # D. Info State: 1 = Drifting (Momentum Persistence)
    y_info = y_labels['info_regime'].apply(lambda x: 1 if x == InfoRegime.DRIFTING else 0)
    
    print("\nTarget Balances (1 = Active):")
    print(f"Vol (Risk Off): {y_vol.mean():.2%}")
    print(f"Trend (Robust): {y_trend.mean():.2%}")
    print(f"Liq (Stressed): {y_liq.mean():.2%}")
    print(f"Info (Drifting): {y_info.mean():.2%}")
    
    # 4. Train/Test Split (Simple Walk Forward Proxy)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    targets = {
        'Vol': (y_vol[:split_idx], y_vol[split_idx:]),
        'Trend': (y_trend[:split_idx], y_trend[split_idx:]),
        'Liq': (y_liq[:split_idx], y_liq[split_idx:]),
        'Info': (y_info[:split_idx], y_info[split_idx:])
    }
    
    models = {}
    preds = {}
    
    print("\nTraining Models...")
    for name, (y_train, y_test) in targets.items():
        print(f"\n--- Model {name} ---")
        print(f"Test Set Class 1 Balance: {y_test.mean():.2%}")
        
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)
        
        preds_series = pd.Series(p)
        print(f"Predictions Dist: {preds_series.value_counts().to_dict()}")
        
        acc = accuracy_score(y_test, p)
        print(f"Accuracy: {acc:.2%}")
        
        models[name] = clf
        preds[name] = p
        
    # 5. Orthogonality Check (Correlation of Predictions)
    print("\nOrthogonality Check (Correlation Matrix of Predictions):")
    pred_df = pd.DataFrame(preds)
    corr = pred_df.corr()
    print(corr)
    
    # Check max correlation
    np.fill_diagonal(corr.values, 0) # Ignore diagonal
    max_corr = corr.abs().max().max()
    print(f"\nMax Inter-Regime Correlation: {max_corr:.2f}")
    
    if max_corr < 0.7:
        print("PASS: Regimes are sufficiently orthogonal.")
    else:
        print("FAIL: Regimes are too correlated.")

if __name__ == "__main__":
    train_and_evaluate()
