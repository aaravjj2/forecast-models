#!/usr/bin/env python3
"""
Investigate XGBoost Feature Importance - Find Leakage Source
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent
env_file = project_root / 'keys.env'
if env_file.exists():
    load_dotenv(env_file)

sys.path.insert(0, str(project_root / 'src'))
os.chdir(project_root)

import pandas as pd
import numpy as np
from models import XGBoostModel
from utils.config import PROCESSED_DATA_DIR

print("="*70)
print("INVESTIGATING XGBOOST FEATURE IMPORTANCE")
print("="*70)

TICKER = "AAPL"

# Load features
features = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_features.csv", index_col=0, parse_dates=True)

feature_cols = [col for col in features.columns 
                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
X = features[feature_cols].fillna(0)
y_direction = features['target_direction']

# Remove neutral targets
mask = y_direction != 0
X_filtered = X[mask]
y_filtered = y_direction[mask]

print(f"\nTraining XGBoost on {len(X_filtered)} samples with {len(feature_cols)} features")

# Train XGBoost
model = XGBoostModel(min_confidence=0.0)
metrics = model.train(X_filtered, y_filtered)
print(f"\nTraining Metrics: {metrics}")

# Get feature importance
importance = model.model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance.items()
]).sort_values('importance', ascending=False)

print("\n" + "="*70)
print("TOP 20 MOST IMPORTANT FEATURES (by gain)")
print("="*70)
print(importance_df.head(20).to_string(index=False))

# Check for suspicious features
print("\n" + "="*70)
print("CHECKING FOR SUSPICIOUS FEATURES")
print("="*70)

# Look for features that might contain future information
suspicious_patterns = ['close', 'high', 'low', 'return_1d', 'momentum_1']
for pattern in suspicious_patterns:
    matching = [f for f in feature_cols if pattern in f.lower()]
    if matching:
        print(f"\nFeatures containing '{pattern}':")
        for f in matching:
            imp = importance.get(f, 0)
            print(f"  {f}: importance={imp:.2f}")

# Test with ONLY lagged features (no same-day data)
print("\n" + "="*70)
print("TESTING WITH ONLY LAGGED FEATURES (>= 2 days)")
print("="*70)

# Keep only features with lookback >= 2 days
safe_features = []
for col in feature_cols:
    # Keep if it has a number >= 2 in it (e.g., return_2d, ma_5, etc.)
    # Exclude anything with _1d or _1 that might be same-day
    if '_1d' not in col and '_1' not in col.split('_')[-1]:
        safe_features.append(col)
    # Also keep if explicitly multi-day
    elif any(f'_{d}d' in col for d in [2, 3, 5, 7, 10, 14, 20, 30, 50]):
        safe_features.append(col)

print(f"\nSafe features: {len(safe_features)} out of {len(feature_cols)}")
print(f"Excluded features: {set(feature_cols) - set(safe_features)}")

X_safe = X_filtered[safe_features]
model_safe = XGBoostModel(min_confidence=0.0)
metrics_safe = model_safe.train(X_safe, y_filtered)
print(f"\nSafe Model Metrics: {metrics_safe}")
