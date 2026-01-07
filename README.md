# ML Research Pipeline for Stock Prediction

A modular, free-tier-compatible ML research pipeline for stock price prediction using ensemble methods with abstention.

## Project Structure

```
ml_research_pipeline/
├── notebooks/          # Jupyter notebooks (00-05)
├── src/               # Source code modules
│   ├── data/         # Data fetching modules
│   ├── features/     # Feature engineering
│   ├── models/       # Specialist models
│   ├── ensemble/     # Meta-gating ensemble
│   ├── backtest/     # Walk-forward backtesting
│   └── utils/        # Utilities
├── tests/            # Unit and integration tests
├── data/             # Data storage
├── models/           # Saved models
├── results/          # Backtest results
└── artifacts/        # All outputs
```

## Setup

### Colab Setup
1. Upload this entire folder to Colab
2. Upload `keys.env` to `/content/ml_research_pipeline/`
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order (00 → 05)

### Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables from `keys.env`
3. Run notebooks in order

## Notebooks

1. **00_data_fetch.ipynb** - Download historical prices and news
2. **01_feature_engineering.ipynb** - Build features (returns, volatility, technical indicators, news)
3. **02_train_price_models.ipynb** - Train XGBoost and LightGBM
4. **03_train_sentiment_models.ipynb** - Train sentiment-based models
5. **04_train_meta_ensemble.ipynb** - Train meta-gating ensemble
6. **05_walkforward_backtest.ipynb** - Walk-forward backtesting and evaluation

## Key Features

- **Free-tier compatible**: Works on Colab/Kaggle free tiers
- **Modular design**: Each component is independent
- **Ensemble with abstention**: Only trades when confident
- **Walk-forward testing**: Realistic evaluation without look-ahead bias
- **Reproducible**: All artifacts saved for later analysis

## Environment Variables

Required API keys (from `keys.env`):
- `FINNHUB_API_KEY` - For news data
- `NEWS_API_KEY` - Alternative news source
- `TIINGO_API_KEY` - Alternative price data (optional)

## Usage

Run notebooks sequentially. Each notebook saves artifacts that the next one uses.

## Testing

Run tests: `pytest tests/`

Run end-to-end test: `python tests/test_e2e.py`


