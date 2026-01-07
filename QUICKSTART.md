# Quick Start Guide

## Installation

### On Google Colab

1. Upload the entire `ml_research_pipeline` folder to Colab
2. Upload `keys.env` to the project root
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Run notebooks in order (00 → 05)

### Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables from `keys.env`
3. Run notebooks in order

## Running the Pipeline

### Step 1: Fetch Data (Notebook 00)
- Downloads historical prices and news
- Caches data locally
- Saves to `data/processed/`

### Step 2: Build Features (Notebook 01)
- Creates time-aligned features
- No data leakage
- Saves features and metadata

### Step 3: Train Price Models (Notebook 02)
- Trains XGBoost and LightGBM
- Saves models to `models/specialists/`

### Step 4: Train Sentiment Models (Notebook 03)
- Trains sentiment and rule-based models
- Saves models to `models/specialists/`

### Step 5: Train Meta-Ensemble (Notebook 04)
- Combines all specialists
- Learns when to trust each model
- Saves to `models/meta/`

### Step 6: Walk-Forward Backtest (Notebook 05)
- Runs realistic backtesting
- Generates metrics and visualizations
- Saves results to `results/`

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run end-to-end test:
```bash
python tests/test_e2e.py
```

## Configuration

Edit `src/utils/config.py` to customize:
- Training window sizes
- Confidence thresholds
- Backtest parameters

## Troubleshooting

**Import errors**: Make sure you're running from the project root and `src/` is in Python path.

**No data**: Check API keys in `keys.env` or Colab secrets.

**Out of memory**: Reduce training window size or use smaller models.

**Slow training**: Reduce `n_estimators` in model parameters.



