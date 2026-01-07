# Run Pipeline Locally

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm yfinance requests python-dotenv tqdm joblib transformers torch sentencepiece finnhub
```

### 2. Ensure keys.env is in place

Make sure `keys.env` is in the project root with all your API keys.

### 3. Run the Pipeline

```bash
cd ml_research_pipeline
python run_pipeline_local.py
```

That's it! The script will:
- ✅ Load all keys from `keys.env`
- ✅ Fetch data (prices + news from multiple sources)
- ✅ Build features
- ✅ Train all models
- ✅ Run backtest
- ✅ Save all results

## News Sources Priority

The pipeline now prioritizes news sources to avoid NewsAPI limits:

1. **Finnhub** (Primary) - More reliable, better free tier (60 calls/min)
2. **NewsAPI** (Fallback) - Only used if Finnhub data is insufficient or NewsAPI limit not reached

If NewsAPI limit is reached, the pipeline continues with Finnhub data only.

## Configuration

Edit `run_pipeline_local.py` to change:
- `TICKER`: Stock symbol (default: "AAPL")
- `START_DATE`: Start date (default: "2020-01-01")
- `END_DATE`: End date (default: "2023-12-31")
- `INDEX_SYMBOL`: Market index (default: "^GSPC" for S&P 500)

## Output

All results are saved to:
- `data/processed/`: Processed data and features
- `models/specialists/`: Trained specialist models
- `models/meta/`: Meta-ensemble model
- `results/`: Backtest results and predictions

## Troubleshooting

**Import errors:**
- Make sure you're in the project root directory
- Check that `src/` directory exists with all modules

**API key errors:**
- Verify `keys.env` exists and has all required keys
- Check that keys are valid (especially FINNHUB_API_KEY)

**NewsAPI limit:**
- This is expected - the pipeline will use Finnhub data only
- NewsAPI is only a fallback source

