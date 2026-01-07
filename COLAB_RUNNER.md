# Running the Pipeline on Colab

## Quick Start

### Option 1: Using Colab Extension from Cursor (Recommended)

1. **Connect to Colab:**
   - In Cursor, press `Ctrl+Shift+P`
   - Type "Colab: Connect to Colab"
   - Authenticate and select a runtime
   - Select Colab kernel in your notebook (top-right)

2. **Add Secrets to Colab:**
   - Open Google Colab in browser
   - Click 🔑 icon (left sidebar)
   - Go to "Secrets" tab
   - Add these keys:
     - `FINNHUB_API_KEY`: d28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660
     - `NEWS_API_KEY`: 9ff201f1e68b4544ab5d358a261f1742
     - `TIINGO_API_KEY`: b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f

3. **Upload Project:**
   - Upload entire `ml_research_pipeline` folder to Colab
   - Or use: `!git clone <your-repo>` if you have it in a repo

4. **Run Notebooks:**
   - Open notebooks in Cursor (they'll run on Colab)
   - Run 00-05 in sequence
   - Or use the master runner notebook

### Option 2: Direct Colab Setup

1. **Open Google Colab**
2. **Run setup cell:**
   ```python
   # Copy and paste colab_auto_setup.py content
   # This will install dependencies and configure everything
   ```
3. **Add secrets** (if not already added)
4. **Upload project files**
5. **Run notebooks 00-05**

## Automated Setup Script

Run this in Colab to set up everything automatically:

```python
# Install dependencies
!pip install -q pandas numpy scikit-learn xgboost lightgbm yfinance requests python-dotenv tqdm joblib transformers torch sentencepiece

# Set up project
import os
from pathlib import Path
project_dir = Path('/content/ml_research_pipeline')
project_dir.mkdir(exist_ok=True)

# Set API keys (from secrets or directly)
from google.colab import userdata
import os

# Get from secrets or use defaults
try:
    os.environ['FINNHUB_API_KEY'] = userdata.get('FINNHUB_API_KEY')
except:
    os.environ['FINNHUB_API_KEY'] = 'd28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660'

try:
    os.environ['NEWS_API_KEY'] = userdata.get('NEWS_API_KEY')
except:
    os.environ['NEWS_API_KEY'] = '9ff201f1e68b4544ab5d358a261f1742'

try:
    os.environ['TIINGO_API_KEY'] = userdata.get('TIINGO_API_KEY')
except:
    os.environ['TIINGO_API_KEY'] = 'b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f'

print("✓ Setup complete")
```

## Running the Pipeline

Once set up, run notebooks in order:

1. `00_data_fetch.ipynb` - Download data
2. `01_feature_engineering.ipynb` - Build features  
3. `02_train_price_models.ipynb` - Train XGBoost/LightGBM
4. `03_train_sentiment_models.ipynb` - Train sentiment models
5. `04_train_meta_ensemble.ipynb` - Train ensemble
6. `05_walkforward_backtest.ipynb` - Backtest

## Notes

- All notebooks automatically detect Colab environment
- Keys are loaded from Colab secrets if available
- Falls back to keys.env or environment variables
- Results are saved to `/content/ml_research_pipeline/results/`


