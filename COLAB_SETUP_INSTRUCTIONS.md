# Colab Setup Instructions

## Quick Setup (5 minutes)

### Step 1: Connect Cursor to Colab

1. In Cursor, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: `Colab: Connect to Colab`
3. Authenticate with your Google account
4. Select a Colab runtime
5. Open `MASTER_RUNNER_COLAB.ipynb` in Cursor
6. Click the kernel selector (top-right) and choose your Colab runtime

### Step 2: Add Secrets to Colab

**Option A: Via Colab UI (Recommended)**
1. Open Google Colab in your browser
2. Click the 🔑 icon (left sidebar)
3. Go to "Secrets" tab
4. Click "Add new secret" for each:
   - **Key**: `FINNHUB_API_KEY`, **Value**: `d28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660`
   - **Key**: `NEWS_API_KEY`, **Value**: `9ff201f1e68b4544ab5d358a261f1742`
   - **Key**: `TIINGO_API_KEY`, **Value**: `b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f`

**Option B: Programmatically (in Colab)**
Run this in a Colab cell:
```python
from google.colab import userdata
# Note: Secrets must be added via UI first, then accessed via userdata.get()
```

### Step 3: Upload Project to Colab

**Option A: Direct Upload**
1. In Colab, click folder icon (left sidebar)
2. Click "Upload" 
3. Upload entire `ml_research_pipeline` folder
4. Or zip it first, upload zip, then unzip: `!unzip ml_research_pipeline.zip`

**Option B: From GitHub**
```python
!git clone <your-repo-url>
```

**Option C: From Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy from drive to /content
```

### Step 4: Run the Pipeline

**In Cursor (with Colab extension):**
1. Open `MASTER_RUNNER_COLAB.ipynb`
2. Make sure Colab kernel is selected
3. Run all cells (Shift+Enter through each)

**Or in Colab directly:**
1. Upload `MASTER_RUNNER_COLAB.ipynb` to Colab
2. Run all cells

## What Gets Executed

The master notebook will:
1. ✅ Install all dependencies
2. ✅ Set up API keys from Colab secrets
3. ✅ Create project structure
4. ✅ Fetch data (prices + news)
5. ✅ Build features
6. ✅ Train all specialist models
7. ✅ Train meta-ensemble
8. ✅ Run walk-forward backtest
9. ✅ Save all results

## Expected Runtime

- **Data fetching**: 2-5 minutes (depends on API rate limits)
- **Feature engineering**: 30 seconds
- **Model training**: 5-10 minutes (XGBoost + LightGBM)
- **Sentiment models**: 1-2 minutes
- **Meta-ensemble**: 2-3 minutes
- **Backtesting**: 5-10 minutes

**Total**: ~20-30 minutes for full pipeline

## Troubleshooting

**"Module not found"**: Make sure project is uploaded and `src/` is in path

**"API key not found"**: Add secrets via Colab UI (🔑 → Secrets)

**"Out of memory"**: Use smaller training windows or reduce model complexity

**"Connection timeout"**: Reconnect to Colab runtime

## Results Location

All results saved to:
- `/content/ml_research_pipeline/results/` (on Colab)
- Or `results/` directory locally

Files created:
- `{TICKER}_backtest_results.pkl` - Full results
- `{TICKER}_predictions.csv` - All predictions
- `{TICKER}_actuals.csv` - Actual returns
- Models saved in `models/` directory



