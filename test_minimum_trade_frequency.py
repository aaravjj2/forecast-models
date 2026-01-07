#!/usr/bin/env python3
"""
Test 8: Minimum Trade Frequency Constraint
Force the model to take responsibility by requiring minimum trades
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from backtest import WalkForwardBacktest
from ensemble import MetaEnsemble
from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel
from utils.config import PROCESSED_DATA_DIR, RESULTS_DIR

def test_minimum_trade_frequency(ticker="AAPL", min_trades_per_period=5):
    """Test with minimum trade frequency constraint"""
    
    print("="*70)
    print("PHASE 4.2: MINIMUM TRADE FREQUENCY TEST")
    print("="*70)
    print(f"Requiring minimum {min_trades_per_period} trades per test period")
    
    # Load data
    features = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(PROCESSED_DATA_DIR / f"{ticker}_prices.csv", index_col=0, parse_dates=True)
    
    feature_cols = [col for col in features.columns 
                    if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
    X = features[feature_cols].fillna(0)
    y = features['target_return_1d']
    y_dir = features['target_direction']
    
    mask = y_dir != 0
    X_train = X[mask]
    y_train = y_dir[mask]
    
    # Train models
    xgb = XGBoostModel(min_confidence=0.6)
    lgb = LightGBMModel(min_confidence=0.6)
    sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
    rule = RuleBasedModel(min_confidence=0.6)
    
    xgb.train(X_train, y_train)
    lgb.train(X_train, y_train)
    sentiment.train(X_train, y_train)
    rule.train(X_train, y_train)
    
    specialists = [xgb, lgb, sentiment, rule]
    
    vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
    news_cols = [col for col in X.columns if 'news' in col.lower()]
    market_features = pd.DataFrame(index=X.index)
    if vol_cols:
        market_features['volatility'] = X[vol_cols[0]]
    if news_cols:
        market_features['news_intensity'] = X[news_cols[0]]
    market_features = market_features.fillna(0)
    
    ensemble = MetaEnsemble(specialists, min_confidence=0.6)
    ensemble.train(X_train, y_train, market_features)
    
    # Modified backtest with minimum trade frequency
    class ConstrainedBacktest(WalkForwardBacktest):
        def __init__(self, ensemble, train_window_days=252, test_window_days=21, min_trades=5):
            super().__init__(ensemble, train_window_days, test_window_days)
            self.min_trades = min_trades
        
        def run(self, X, prices, y):
            results = super().run(X, prices, y)
            
            # Adjust confidence threshold to meet minimum trades
            predictions = results['predictions']
            test_periods = len(predictions) // self.test_window_days
            
            for period in range(test_periods):
                period_start = period * self.test_window_days
                period_end = min((period + 1) * self.test_window_days, len(predictions))
                period_preds = predictions.iloc[period_start:period_end]
                
                trades = (period_preds['signal'] != 0).sum()
                if trades < self.min_trades:
                    # Lower threshold for this period
                    period_preds_sorted = period_preds[period_preds['signal'] == 0].sort_values('confidence', ascending=False)
                    needed = self.min_trades - trades
                    if len(period_preds_sorted) >= needed:
                        # Convert top abstentions to trades
                        for idx in period_preds_sorted.head(needed).index:
                            if period_preds.loc[idx, 'prob_up'] > 0.5:
                                predictions.loc[idx, 'signal'] = 1
                            else:
                                predictions.loc[idx, 'signal'] = -1
            
            # Recalculate metrics
            from backtest.walkforward_backtest import calculate_metrics
            results['metrics'] = calculate_metrics(results['predictions'], results['actuals'], prices)
            
            return results
    
    backtest = ConstrainedBacktest(ensemble, train_window_days=252, test_window_days=21, min_trades=min_trades_per_period)
    results = backtest.run(X, prices, y)
    
    metrics = results['metrics']
    predictions = results['predictions']
    
    trades = (predictions['signal'] != 0).sum()
    
    print(f"\nResults with minimum trade frequency:")
    print(f"  Total trades: {trades}")
    print(f"  Accuracy: {metrics.get('directional_accuracy', 0):.4f}")
    print(f"  Total return: {metrics.get('total_return', 0):.4f}")
    print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Coverage: {metrics.get('coverage', 0):.2%}")
    
    return {
        'min_trades_period': min_trades_per_period,
        'total_trades': int(trades),
        'accuracy': float(metrics.get('directional_accuracy', 0)),
        'total_return': float(metrics.get('total_return', 0)),
        'sharpe': float(metrics.get('sharpe_ratio', 0)),
        'coverage': float(metrics.get('coverage', 0))
    }

if __name__ == "__main__":
    result = test_minimum_trade_frequency()
    print(f"\n✓ Test complete: {result}")


