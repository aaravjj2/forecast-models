#!/usr/bin/env python3
"""
Local Pipeline Runner - Run the entire ML research pipeline locally.

This script runs the complete pipeline without needing Colab.
All API keys should be in keys.env file.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent
# Try keys.env in project root first, then parent directory
env_file = project_root / 'keys.env'
if not env_file.exists():
    env_file = project_root.parent / 'keys.env'
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded keys from {env_file}")
else:
    print("⚠ keys.env not found - using environment variables")

# Add src to path
sys.path.insert(0, str(project_root / 'src'))

# Change to project directory
os.chdir(project_root)

import pandas as pd
from data import PriceFetcher, NewsFetcher
from features import FeatureBuilder
from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel
from ensemble import MetaEnsemble
from backtest import WalkForwardBacktest
from utils.config import (
    PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR, META_MODEL_DIR, RESULTS_DIR
)
from utils.helpers import save_artifact

print("="*70)
print("ML RESEARCH PIPELINE - LOCAL EXECUTION")
print("="*70)
print(f"Project directory: {project_root}")
print(f"Working directory: {os.getcwd()}\n")

# Configuration
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
INDEX_SYMBOL = "^GSPC"

# ============================================================================
# STEP 1: Data Fetching
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA FETCHING")
print("="*70)

print(f"Fetching data for {TICKER}...")

# Fetch prices
price_fetcher = PriceFetcher()
stock_prices = price_fetcher.fetch(TICKER, START_DATE, END_DATE)
print(f"✓ Fetched {len(stock_prices)} days of price data")

# Fetch index
index_prices = price_fetcher.fetch_index(INDEX_SYMBOL, START_DATE, END_DATE)
print(f"✓ Fetched {len(index_prices)} days of index data")

# Fetch news (using multiple sources, not just NewsAPI)
news_fetcher = NewsFetcher()
news_data = news_fetcher.fetch_all(TICKER, START_DATE, END_DATE)
print(f"✓ Fetched {len(news_data)} news articles from multiple sources")

# Save
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
stock_prices.to_csv(PROCESSED_DATA_DIR / f"{TICKER}_prices.csv")
index_prices.to_csv(PROCESSED_DATA_DIR / f"{INDEX_SYMBOL.replace('^', '')}_prices.csv")
if not news_data.empty:
    news_data.to_csv(PROCESSED_DATA_DIR / f"{TICKER}_news.csv", index=False)

print("\n✓ Data fetching complete")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

# Load data
stock_prices = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_prices.csv", index_col=0, parse_dates=True)
index_prices = pd.read_csv(PROCESSED_DATA_DIR / f"{INDEX_SYMBOL.replace('^', '')}_prices.csv", index_col=0, parse_dates=True)

news_file = PROCESSED_DATA_DIR / f"{TICKER}_news.csv"
if news_file.exists():
    news_data = pd.read_csv(news_file, parse_dates=['date'])
else:
    news_data = pd.DataFrame()

# Build features
builder = FeatureBuilder()
features = builder.build_all_features(stock_prices, index_prices, news_data)

print(f"✓ Built {len(features.columns)} features, {len(features)} samples")

# Save
features.to_csv(PROCESSED_DATA_DIR / f"{TICKER}_features.csv")
save_artifact(builder.feature_metadata, PROCESSED_DATA_DIR / f"{TICKER}_feature_metadata.pkl")

print("✓ Feature engineering complete")

# ============================================================================
# STEP 3: Train Price Models
# ============================================================================
print("\n" + "="*70)
print("STEP 3: TRAIN PRICE MODELS")
print("="*70)

# Load features
features = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_features.csv", index_col=0, parse_dates=True)

feature_cols = [col for col in features.columns 
                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
X = features[feature_cols].fillna(0)
y = features['target_direction']
mask = y != 0
X = X[mask]
y = y[mask]

print(f"Training on {len(X)} samples")

# Train XGBoost
print("\nTraining XGBoost...")
xgb = XGBoostModel(min_confidence=0.6)
xgb_metrics = xgb.train(X, y)
print(f"✓ XGBoost: {xgb_metrics}")
SPECIALIST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
xgb.save(SPECIALIST_MODELS_DIR / f"{TICKER}_xgb.model")

# Train LightGBM
print("\nTraining LightGBM...")
lgb = LightGBMModel(min_confidence=0.6)
lgb_metrics = lgb.train(X, y)
print(f"✓ LightGBM: {lgb_metrics}")
lgb.save(SPECIALIST_MODELS_DIR / f"{TICKER}_lgb.txt")

print("\n✓ Price models trained")

# ============================================================================
# STEP 4: Train Sentiment Models
# ============================================================================
print("\n" + "="*70)
print("STEP 4: TRAIN SENTIMENT MODELS")
print("="*70)

# Train Sentiment
print("\nTraining Sentiment Model...")
sentiment = SentimentModel(min_confidence=0.6, use_pretrained=False)
sentiment_metrics = sentiment.train(X, y)
print(f"✓ Sentiment: {sentiment_metrics}")
sentiment.save(SPECIALIST_MODELS_DIR / f"{TICKER}_sentiment.pkl")

# Train Rule-based
print("\nTraining Rule-based Model...")
rule = RuleBasedModel(min_confidence=0.6)
rule_metrics = rule.train(X, y)
print(f"✓ Rule-based: {rule_metrics}")
rule.save(SPECIALIST_MODELS_DIR / f"{TICKER}_rule.pkl")

print("\n✓ Sentiment models trained")

# ============================================================================
# STEP 5: Train Meta-Ensemble
# ============================================================================
print("\n" + "="*70)
print("STEP 5: TRAIN META-ENSEMBLE")
print("="*70)

# Load all specialists
print("Loading specialist models...")
xgb = XGBoostModel()
xgb.load(SPECIALIST_MODELS_DIR / f"{TICKER}_xgb.model")

lgb = LightGBMModel()
lgb.load(SPECIALIST_MODELS_DIR / f"{TICKER}_lgb.txt")

sentiment = SentimentModel()
sentiment.load(SPECIALIST_MODELS_DIR / f"{TICKER}_sentiment.pkl")

rule = RuleBasedModel()
rule.load(SPECIALIST_MODELS_DIR / f"{TICKER}_rule.pkl")

specialists = [xgb, lgb, sentiment, rule]

# Extract market features
vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
news_cols = [col for col in X.columns if 'news' in col.lower()]

market_features = pd.DataFrame(index=X.index)
if vol_cols:
    market_features['volatility'] = X[vol_cols[0]]
if news_cols:
    market_features['news_intensity'] = X[news_cols[0]]
market_features = market_features.fillna(0)

# Train ensemble
print("\nTraining Meta-Ensemble...")
ensemble = MetaEnsemble(specialists, min_confidence=0.6)
meta_metrics = ensemble.train(X, y, market_features)
print(f"✓ Meta-ensemble: {meta_metrics}")
META_MODEL_DIR.mkdir(parents=True, exist_ok=True)
ensemble.save(META_MODEL_DIR / f"{TICKER}_meta_ensemble.pkl")

print("\n✓ Meta-ensemble trained")

# ============================================================================
# STEP 6: Walk-Forward Backtest
# ============================================================================
print("\n" + "="*70)
print("STEP 6: WALK-FORWARD BACKTEST")
print("="*70)

# Load data
features = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_features.csv", index_col=0, parse_dates=True)
prices = pd.read_csv(PROCESSED_DATA_DIR / f"{TICKER}_prices.csv", index_col=0, parse_dates=True)

feature_cols = [col for col in features.columns 
                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]
X = features[feature_cols].fillna(0)
y = features['target_return_1d']

# Load ensemble
print("Loading ensemble...")
ensemble = MetaEnsemble(specialists)
ensemble.load(META_MODEL_DIR / f"{TICKER}_meta_ensemble.pkl")

# Run backtest
print("\nRunning walk-forward backtest...")
backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)
results = backtest.run(X, prices, y)

print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)
for key, value in results['metrics'].items():
    print(f"{key}: {value:.4f}")

# Save
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
backtest.save_results(RESULTS_DIR / f"{TICKER}_backtest_results.pkl")
results['predictions'].to_csv(RESULTS_DIR / f"{TICKER}_predictions.csv")
results['actuals'].to_csv(RESULTS_DIR / f"{TICKER}_actuals.csv")

print("\n✓ Backtest complete - results saved to results/")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
print(f"✓ All models saved to: {SPECIALIST_MODELS_DIR}")
print(f"✓ Ensemble saved to: {META_MODEL_DIR}")
print(f"✓ Results saved to: {RESULTS_DIR}")
print(f"✓ Features saved to: {PROCESSED_DATA_DIR}")
print("\nPipeline execution completed successfully!")

