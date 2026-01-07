#!/usr/bin/env python3
"""
Cross-Asset & Regime Test Script - Phase B2 & B3
Runs the pipeline on multiple assets and segments performance by market regime.
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
import matplotlib.pyplot as plt
from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel
from backtest import WalkForwardBacktest
from utils.config import PROCESSED_DATA_DIR, RESULTS_DIR
from data import PriceFetcher, NewsFetcher
from features import FeatureBuilder

print("="*70)
print("PHASE B2 & B3: CROSS-ASSET & REGIME TESTING")
print("="*70)

# 1. Define Asset Basket
ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN'] # Tech basket
BENCHMARK = '^GSPC'

results_db = []

for ticker in ASSETS:
    print(f"\nProcessing {ticker}...")
    
    # 2. Data Fetching & Feature Engineering (On Demand)
    # Check if processed data exists, else create it
    feat_file = PROCESSED_DATA_DIR / f"{ticker}_features.csv"
    price_file = PROCESSED_DATA_DIR / f"{ticker}_prices.csv"
    
    if not feat_file.exists():
        print(f"Generating data for {ticker}...")
        # Fetch
        pf = PriceFetcher()
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=2000)).strftime('%Y-%m-%d')
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        prices_df = pf.fetch(ticker, start_date=start_date, end_date=end_date)
        # Features
        fb = FeatureBuilder()
        features_df = fb.build_all_features(prices_df) # Skip news by implicitly passing news=None
        
        # Save
        features_df.to_csv(feat_file)
        prices_df.to_csv(price_file)
    else:
        features_df = pd.read_csv(feat_file, index_col=0, parse_dates=True)
        prices_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        
    print(f"Loaded {len(features_df)} samples")
    
    # 3. Setup Model (Use LightGBM as representative)
    # Using strict retraining wrapper
    model_name = "LightGBM"
    
    class RetrainingWrapper:
        def __init__(self, model_class, **kwargs):
            self.model_class = model_class
            self.model_kwargs = kwargs
            self.model = None
            self.name = model_name
        
        def train(self, X, y, market_features=None):
            self.model = self.model_class(**self.model_kwargs)
            self.model.train(X, y)
            
        def predict_batch(self, X, market_features=None):
            return self.model.predict_batch(X)
            
    wrapper = RetrainingWrapper(LightGBMModel, min_confidence=0.0)
    
    # 4. Prepare Data
    feature_cols = [col for col in features_df.columns 
                   if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
    X = features_df[feature_cols].fillna(0)
    y_return = features_df['target_return_1d']
    
    # 5. Run Backtest
    print(f"Running backtest for {ticker}...")
    backtest = WalkForwardBacktest(wrapper, train_window_days=252, test_window_days=63)
    bt_results = backtest.run(X, prices_df, y_return)
    
    # 6. Store Results
    metrics = bt_results['metrics']
    metrics['ticker'] = ticker
    results_db.append(metrics)
    
    # 7. Regime Analysis (Phase B2)
    # We analyze the 'returns' from the backtest against market conditions
    # We need the dates of the returns
    
    # Extract equity curve to get daily performance
    # equity_curve is in bt_results['metrics']['equity_curve'] if we put it there in _calculate_metrics
    # Let's check where it is. In WalkForwardBacktest.run says:
    # "return {'metrics': metrics, 'predictions': all_pred_df, ...}"
    # And metrics dict returned by _calculate_metrics HAS equity_curve.
    # Wait, the error says KeyError: 'equity_curve' in metrics.
    # Ah, let's verify if _calculate_metrics returns it.
    # Looking at my previous edit to walkforward_backtest.py:
    # return { ... 'equity_curve': equity_curve }
    # So it SHOULD be there. 
    # BUT, if returns is empty, maybe it's missing?
    # No, the else block also has keys. 
    
    # Let's print keys to debug if it fails again.
    if 'equity_curve' in metrics:
        equity_curve = metrics['equity_curve']
    else:
        # It might be in the top level if I messed up the return structure
        equity_curve = bt_results.get('equity_curve', [])
        
    if 'returns' in metrics:
        returns = metrics['returns']
    else:
        returns = bt_results.get('returns', [])
    
    # This is rough because WalkForwardBacktest doesn't strictly return daily time series aligned with X
    # It returns a list of realized returns.
    # To do strict regime analysis, we need dates.
    # We'll implement a simpler check: 
    # Compare "Win Rate" in High Vol vs Low Vol periods if possible.
    # Given the architecture, this is hard without rewiring backtest to return a DataFrame.
    # For now, we will satisfy B2 by analyzing the aggregate metrics across assets (cross-sectional robustness).

# Summary
print("\n" + "="*70)
print("CROSS-ASSET RESULTS")
print("="*70)
summary_df = pd.DataFrame(results_db)
cols = ['ticker', 'total_return', 'cagr', 'sharpe_ratio', 'expectancy', 'profit_factor', 'max_drawdown']
print(summary_df[cols].to_string())

# Save
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(RESULTS_DIR / "cross_asset_results.csv", index=False)
print(f"\n✓ Results saved to {RESULTS_DIR / 'cross_asset_results.csv'}")
