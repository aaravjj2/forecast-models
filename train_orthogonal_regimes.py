#!/usr/bin/env python3
"""
Train Orthogonal Regimes - Phase B
Trains independent classifiers for:
1. Volatility Regime (Risk Off vs Risk On)
2. Trend Quality Regime (Robust vs Noisy)

And tests for prediction correlation (Orthogonality).
"""

import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality

def train_and_evaluate_orthogonality(ticker='AAPL'):
    print(f"=== Orthogonality Testing for {ticker} ===")
    
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
    
    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y_regime = labels.loc[common_idx, 'regime']
    y_quality = labels.loc[common_idx, 'trend_quality']
    
    # Clean X
    drop_cols = ['next_day_return', 'target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume', 'regime', 'regime_name', 'trend_quality', 'trend_quality_name']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # 2. Define Targets
    # Target 1: Volatility (Risk Off)
    # 1 = High Vol/Crash, 0 = Normal
    y_vol = y_regime.apply(lambda x: 1 if x in [MarketRegime.HIGH_VOL, MarketRegime.CRASH] else 0)
    
    # Target 2: Trend Quality (Robust)
    # 1 = Robust Trend, 0 = Noisy/Neutral
    y_trend = y_quality.apply(lambda x: 1 if x == TrendQuality.ROBUST else 0) 
    
    print("\nClass Balance (Trend Quality 1=Robust):")
    print(y_trend.value_counts(normalize=True))
    
    # 3. Train Independent Models (Walk Forward/CV)
    print("\nTraining Independent Classifiers...")
    
    preds_vol = []
    preds_trend = []
    actuals_vol = []
    actuals_trend = []
    
    # Simple Split for Orthogonality Check
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_vol_train, y_vol_test = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]
    y_trend_train, y_trend_test = y_trend.iloc[:split_idx], y_trend.iloc[split_idx:]
    
    # Model 1: Volatility
    print("Training Volatility Detector...")
    clf_vol = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf_vol.fit(X_train, y_vol_train)
    p_vol = clf_vol.predict(X_test)
    acc_vol = accuracy_score(y_vol_test, p_vol)
    print(f"Vol Accuracy: {acc_vol:.2%}")
    
    # Model 2: Trend Quality
    print("Training Trend Quality Detector...")
    clf_trend = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf_trend.fit(X_train, y_trend_train)
    p_trend = clf_trend.predict(X_test)
    acc_trend = accuracy_score(y_trend_test, p_trend)
    print(f"Trend Accuracy: {acc_trend:.2%}")
    
    # Debug Feature Importance for Trend
    imp = pd.DataFrame({'feat': X.columns, 'gain': clf_trend.feature_importances_}).sort_values('gain', ascending=False)
    print("Top 10 Features for Trend Quality:")
    print(imp.head(10))
    
    # 4. Orthogonality Analysis
    # We want these to be somewhat independent.
    # If Vol=1 (Risk Off) implies Trend=0 (Noisy/Bad), that's fine (correlation).
    # But if they offer *different* information, we gain alpha.
    
    df_res = pd.DataFrame({
        'Vol_Pred': p_vol,
        'Trend_Pred': p_trend
    })
    
    # Correlation (Phi Coefficient / MCC)
    corr = matthews_corrcoef(p_vol, p_trend)
    print(f"\nPrediction Correlation (Phi): {corr:.4f}")
    
    # Overlap Matrix
    # Vol=1 (Risk Off)
    # Trend=1 (Robust Trend)
    # Lattice States:
    # Risk Off (Vol=1) -> Hostile (Strategy Flat)
    # Risk On (Vol=0) + Trend=0 -> Neutral (Strategy MeanRev)
    # Risk On (Vol=0) + Trend=1 -> Favorable (Strategy Momentum)
    
    counts = pd.crosstab(df_res['Vol_Pred'], df_res['Trend_Pred'], rownames=['Risk_Off'], colnames=['Robust_Trend'])
    print("\nOverlap Matrix (Count of Days):")
    print(counts)
    
    normalized = pd.crosstab(df_res['Vol_Pred'], df_res['Trend_Pred'], normalize='all', rownames=['Risk_Off'], colnames=['Robust_Trend'])
    print("\nOverlap Matrix (%):")
    print(normalized)
    
    # Check for excessive duplication
    # If Corr > 0.7 or < -0.7, they are redundant.
    if abs(corr) < 0.7:
        print("\nPASS: Regimes are sufficiently orthogonal.")
    else:
        print("\nFAIL: Regimes are highly correlated. Need to redefine.")
        
    return df_res

if __name__ == "__main__":
    train_and_evaluate_orthogonality()
