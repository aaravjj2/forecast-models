#!/usr/bin/env python3
"""
Test Single Model Baselines - Phase A3
Force models to trade by lowering confidence threshold to 0.0
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent
env_file = project_root / 'keys.env'
if not env_file.exists():
    env_file = project_root.parent / 'keys.env'
if env_file.exists():
    load_dotenv(env_file)

sys.path.insert(0, str(project_root / 'src'))
os.chdir(project_root)

import pandas as pd
import numpy as np
from data import PriceFetcher
from features import FeatureBuilder
from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel
from backtest import WalkForwardBacktest
from utils.config import PROCESSED_DATA_DIR, RESULTS_DIR

print("="*70)
print("PHASE A3: SINGLE MODEL BASELINE TESTING")
print("="*70)

TICKER = "AAPL"

# Load features
features = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_features.csv", index_col=0, parse_dates=True)
prices = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_prices.csv", index_col=0, parse_dates=True)

feature_cols = [col for col in features.columns 
                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
X = features[feature_cols].fillna(0)
y_direction = features['target_direction']
y_return = features['target_return_1d']

# Remove neutral targets
mask = y_direction != 0
X_filtered = X[mask]
y_filtered = y_direction[mask]

print(f"\nTraining on {len(X_filtered)} samples")

# Test each model individually with ZERO confidence threshold
models_to_test = [
    ("XGBoost", XGBoostModel(min_confidence=0.0)),
    ("LightGBM", LightGBMModel(min_confidence=0.0)),
    ("Sentiment", SentimentModel(min_confidence=0.0, use_pretrained=False)),
    ("RuleBased", RuleBasedModel(min_confidence=0.0))
]

results_summary = []

for model_name, model_factory in models_to_test:
    print(f"\n{'='*70}")
    print(f"TESTING: {model_name} (min_confidence=0.0)")
    print("="*70)
    
    # Create single-model "ensemble" that RETRAINS on every window
    class RetrainingWrapper:
        def __init__(self, model_class, **kwargs):
            self.model_class = model_class
            self.model_kwargs = kwargs
            self.model = None
            self.name = model_name
        
        def train(self, X, y, market_features=None):
            # FRESH INSTANTIATION for every window
            self.model = self.model_class(**self.model_kwargs)
            self.model.train(X, y)
            
        def predict_batch(self, X, market_features=None):
            return self.model.predict_batch(X)
            
        def get_signal(self, snapshot, market_snapshot=None):
            return self.model.get_signal(snapshot)
    
    # Extract class and params from the instantiated instances in the list
    # (The list defined above instantiated them, but we need the class to re-instantiate)
    # We'll hack it slightly by grabbing the class from the object
    wrapper = RetrainingWrapper(type(model_factory), min_confidence=0.0, **getattr(model_factory, 'params', {}))
    # Special handling for SentimentModel which has use_pretrained arg in init but not in params dict usually
    if model_name == "Sentiment":
        wrapper = RetrainingWrapper(SentimentModel, min_confidence=0.0, use_pretrained=False)
    
    # Run backtest
    print(f"\nRunning backtest for {model_name}...")
    # PASS UNTRAINED WRAPPER to backtest
    # The backtester calls .train() on it for each window
    backtest = WalkForwardBacktest(wrapper, train_window_days=252, test_window_days=21)
    
    try:
        bt_results = backtest.run(X, prices, y_return)
        
        if 'metrics' in bt_results:
            print(f"\n{model_name} Backtest Results:")
            for key, value in bt_results['metrics'].items():
                print(f"  {key}: {value:.4f}")
            
            results_summary.append({
                'model': model_name,
                **bt_results['metrics']
            })
    except Exception as e:
        print(f"Backtest failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*70)
print("SUMMARY: SINGLE MODEL BASELINES")
print("="*70)

if results_summary:
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "single_model_baselines.csv", index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'single_model_baselines.csv'}")
else:
    print("No results generated")
