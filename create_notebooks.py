"""Script to create remaining notebooks."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"

def create_notebook_01():
    """Create feature engineering notebook."""
    cells = [
        {"cell_type": "markdown", "source": ["# 01: Feature Engineering\n", "\n", "Build time-aligned, leak-free features from price and news data."]},
        {"cell_type": "code", "source": ["# Setup\n", "import sys\n", "from pathlib import Path\n", "import pandas as pd\n", "import numpy as np\n", "\n", "PROJECT_ROOT = Path().absolute().parent.parent\n", "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n", "\n", "from features import FeatureBuilder\n", "from utils.config import PROCESSED_DATA_DIR\n", "\n", "TICKER = \"AAPL\"  # Must match notebook 00"]},
        {"cell_type": "code", "source": ["# Load data\n", "stock_prices = pd.read_csv(PROCESSED_DATA_DIR / f\"{TICKER}_prices.csv\", index_col=0, parse_dates=True)\n", "index_prices = pd.read_csv(PROCESSED_DATA_DIR / \"GSPC_prices.csv\", index_col=0, parse_dates=True)\n", "\n", "# Try loading news\n", "news_file = PROCESSED_DATA_DIR / f\"{TICKER}_news.csv\"\n", "if news_file.exists():\n", "    news_data = pd.read_csv(news_file, parse_dates=['date'])\n", "else:\n", "    news_data = pd.DataFrame()\n", "    print(\"⚠ No news data found\")"]},
        {"cell_type": "code", "source": ["# Build features\n", "builder = FeatureBuilder()\n", "features = builder.build_all_features(stock_prices, index_prices, news_data)\n", "\n", "print(f\"Built {len(features.columns)} features\")\n", "print(f\"Feature columns: {len(builder.feature_metadata['feature_columns'])}\")\n", "print(f\"\\nFeatures shape: {features.shape}\")\n", "print(features.head())"]},
        {"cell_type": "code", "source": ["# Save features and metadata\n", "from utils.helpers import save_artifact\n", "\n", "features.to_csv(PROCESSED_DATA_DIR / f\"{TICKER}_features.csv\")\n", "save_artifact(builder.feature_metadata, PROCESSED_DATA_DIR / f\"{TICKER}_feature_metadata.pkl\")\n", "\n", "print(f\"✓ Saved features and metadata\")"]}
    ]
    return cells

def create_notebook_02():
    """Create price models training notebook."""
    cells = [
        {"cell_type": "markdown", "source": ["# 02: Train Price Models\n", "\n", "Train XGBoost and LightGBM specialist models."]},
        {"cell_type": "code", "source": ["# Setup\n", "import sys\n", "from pathlib import Path\n", "import pandas as pd\n", "\n", "PROJECT_ROOT = Path().absolute().parent.parent\n", "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n", "\n", "from models import XGBoostModel, LightGBMModel\n", "from utils.config import PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR\n", "\n", "TICKER = \"AAPL\""]},
        {"cell_type": "code", "source": ["# Load features\n", "features = pd.read_csv(PROCESSED_DATA_DIR / f\"{TICKER}_features.csv\", index_col=0, parse_dates=True)\n", "\n", "# Prepare data\n", "feature_cols = [col for col in features.columns \n", "                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]\n", "X = features[feature_cols].fillna(0)\n", "y = features['target_direction']\n", "\n", "# Remove rows with zero target (no clear direction)\n", "mask = y != 0\n", "X = X[mask]\n", "y = y[mask]\n", "\n", "print(f\"Training data: {len(X)} samples, {len(feature_cols)} features\")"]},
        {"cell_type": "code", "source": ["# Train XGBoost\n", "xgb_model = XGBoostModel(min_confidence=0.6)\n", "xgb_metrics = xgb_model.train(X, y)\n", "print(f\"XGBoost metrics: {xgb_metrics}\")\n", "\n", "# Save\n", "xgb_model.save(SPECIALIST_MODELS_DIR / f\"{TICKER}_xgb.model\")\n", "print(f\"✓ Saved XGBoost model\")"]},
        {"cell_type": "code", "source": ["# Train LightGBM\n", "lgb_model = LightGBMModel(min_confidence=0.6)\n", "lgb_metrics = lgb_model.train(X, y)\n", "print(f\"LightGBM metrics: {lgb_metrics}\")\n", "\n", "# Save\n", "lgb_model.save(SPECIALIST_MODELS_DIR / f\"{TICKER}_lgb.txt\")\n", "print(f\"✓ Saved LightGBM model\")"]}
    ]
    return cells

def create_notebook_03():
    """Create sentiment models training notebook."""
    cells = [
        {"cell_type": "markdown", "source": ["# 03: Train Sentiment Models\n", "\n", "Train sentiment-based and rule-based models."]},
        {"cell_type": "code", "source": ["# Setup\n", "import sys\n", "from pathlib import Path\n", "import pandas as pd\n", "\n", "PROJECT_ROOT = Path().absolute().parent.parent\n", "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n", "\n", "from models import SentimentModel, RuleBasedModel\n", "from utils.config import PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR\n", "\n", "TICKER = \"AAPL\""]},
        {"cell_type": "code", "source": ["# Load features\n", "features = pd.read_csv(PROCESSED_DATA_DIR / f\"{TICKER}_features.csv\", index_col=0, parse_dates=True)\n", "\n", "# Prepare data\n", "feature_cols = [col for col in features.columns \n", "                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]\n", "X = features[feature_cols].fillna(0)\n", "y = features['target_direction']\n", "\n", "# Remove zero targets\n", "mask = y != 0\n", "X = X[mask]\n", "y = y[mask]\n", "\n", "print(f\"Training data: {len(X)} samples\")"]},
        {"cell_type": "code", "source": ["# Train Sentiment Model\n", "sentiment_model = SentimentModel(min_confidence=0.6, use_pretrained=True)\n", "sentiment_metrics = sentiment_model.train(X, y)\n", "print(f\"Sentiment model metrics: {sentiment_metrics}\")\n", "\n", "# Save\n", "sentiment_model.save(SPECIALIST_MODELS_DIR / f\"{TICKER}_sentiment.pkl\")\n", "print(f\"✓ Saved sentiment model\")"]},
        {"cell_type": "code", "source": ["# Train Rule-Based Model\n", "rule_model = RuleBasedModel(min_confidence=0.6)\n", "rule_metrics = rule_model.train(X, y)\n", "print(f\"Rule-based model metrics: {rule_metrics}\")\n", "\n", "# Save\n", "rule_model.save(SPECIALIST_MODELS_DIR / f\"{TICKER}_rule.pkl\")\n", "print(f\"✓ Saved rule-based model\")"]}
    ]
    return cells

def create_notebook_04():
    """Create meta-ensemble training notebook."""
    cells = [
        {"cell_type": "markdown", "source": ["# 04: Train Meta-Ensemble\n", "\n", "Train meta-gating ensemble that combines specialist models."]},
        {"cell_type": "code", "source": ["# Setup\n", "import sys\n", "from pathlib import Path\n", "import pandas as pd\n", "\n", "PROJECT_ROOT = Path().absolute().parent.parent\n", "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n", "\n", "from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel\n", "from ensemble import MetaEnsemble\n", "from utils.config import PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR, META_MODEL_DIR\n", "\n", "TICKER = \"AAPL\""]},
        {"cell_type": "code", "source": ["# Load all trained specialists\n", "xgb = XGBoostModel()\n", "xgb.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_xgb.model\")\n", "\n", "lgb = LightGBMModel()\n", "lgb.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_lgb.txt\")\n", "\n", "sentiment = SentimentModel()\n", "sentiment.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_sentiment.pkl\")\n", "\n", "rule = RuleBasedModel()\n", "rule.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_rule.pkl\")\n", "\n", "specialists = [xgb, lgb, sentiment, rule]\n", "print(f\"Loaded {len(specialists)} specialist models\")"]},
        {"cell_type": "code", "source": ["# Load features\n", "features = pd.read_csv(PROCESSED_DATA_DIR / f\"{TICKER}_features.csv\", index_col=0, parse_dates=True)\n", "\n", "feature_cols = [col for col in features.columns \n", "                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]\n", "X = features[feature_cols].fillna(0)\n", "y = features['target_direction']\n", "\n", "mask = y != 0\n", "X = X[mask]\n", "y = y[mask]\n", "\n", "print(f\"Training meta-ensemble on {len(X)} samples\")"]},
        {"cell_type": "code", "source": ["# Extract market features for meta-model\n", "vol_cols = [col for col in features.columns if 'volatility' in col.lower()]\n", "news_cols = [col for col in features.columns if 'news' in col.lower()]\n", "\n", "market_features = pd.DataFrame(index=X.index)\n", "if vol_cols:\n", "    market_features['volatility'] = X[vol_cols[0]]\n", "if news_cols:\n", "    market_features['news_intensity'] = X[news_cols[0]]\n", "\n", "market_features = market_features.fillna(0)"]},
        {"cell_type": "code", "source": ["# Train meta-ensemble\n", "ensemble = MetaEnsemble(specialists, min_confidence=0.6)\n", "meta_metrics = ensemble.train(X, y, market_features)\n", "print(f\"Meta-ensemble metrics: {meta_metrics}\")\n", "\n", "# Save\n", "ensemble.save(META_MODEL_DIR / f\"{TICKER}_meta_ensemble.pkl\")\n", "print(f\"✓ Saved meta-ensemble\")"]}
    ]
    return cells

def create_notebook_05():
    """Create walk-forward backtest notebook."""
    cells = [
        {"cell_type": "markdown", "source": ["# 05: Walk-Forward Backtest\n", "\n", "Run walk-forward backtesting and evaluate ensemble performance."]},
        {"cell_type": "code", "source": ["# Setup\n", "import sys\n", "from pathlib import Path\n", "import pandas as pd\n", "import numpy as np\n", "\n", "PROJECT_ROOT = Path().absolute().parent.parent\n", "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n", "\n", "from models import XGBoostModel, LightGBMModel, SentimentModel, RuleBasedModel\n", "from ensemble import MetaEnsemble\n", "from backtest import WalkForwardBacktest\n", "from utils.config import PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR, META_MODEL_DIR, RESULTS_DIR\n", "\n", "TICKER = \"AAPL\""]},
        {"cell_type": "code", "source": ["# Load all models\n", "xgb = XGBoostModel()\n", "xgb.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_xgb.model\")\n", "\n", "lgb = LightGBMModel()\n", "lgb.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_lgb.txt\")\n", "\n", "sentiment = SentimentModel()\n", "sentiment.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_sentiment.pkl\")\n", "\n", "rule = RuleBasedModel()\n", "rule.load(SPECIALIST_MODELS_DIR / f\"{TICKER}_rule.pkl\")\n", "\n", "ensemble = MetaEnsemble([xgb, lgb, sentiment, rule])\n", "ensemble.load(META_MODEL_DIR / f\"{TICKER}_meta_ensemble.pkl\")\n", "\n", "print(\"✓ Loaded all models\")"]},
        {"cell_type": "code", "source": ["# Load data\n", "features = pd.read_csv(PROCESSED_DATA_DIR / f\"{TICKER}_features.csv\", index_col=0, parse_dates=True)\n", "prices = pd.read_csv(PROCESSED_DATA_DIR / f\"{TICKER}_prices.csv\", index_col=0, parse_dates=True)\n", "\n", "feature_cols = [col for col in features.columns \n", "                if col not in ['target_return_1d', 'target_direction', 'open', 'high', 'low', 'close', 'volume']]\n", "X = features[feature_cols].fillna(0)\n", "y = features['target_return_1d']  # Use return for backtesting\n", "\n", "print(f\"Backtest data: {len(X)} samples\")"]},
        {"cell_type": "code", "source": ["# Run walk-forward backtest\n", "backtest = WalkForwardBacktest(ensemble, train_window_days=252, test_window_days=21)\n", "results = backtest.run(X, prices, y)\n", "\n", "print(\"\\n\" + \"=\"*60)\n", "print(\"BACKTEST RESULTS\")\n", "print(\"=\"*60)\n", "for key, value in results['metrics'].items():\n", "    print(f\"{key}: {value:.4f}\")"]},
        {"cell_type": "code", "source": ["# Save results\n", "backtest.save_results(RESULTS_DIR / f\"{TICKER}_backtest_results.pkl\")\n", "results['predictions'].to_csv(RESULTS_DIR / f\"{TICKER}_predictions.csv\")\n", "results['actuals'].to_csv(RESULTS_DIR / f\"{TICKER}_actuals.csv\")\n", "\n", "print(f\"✓ Saved results to {RESULTS_DIR}\")"]},
        {"cell_type": "code", "source": ["# Visualize results\n", "import matplotlib.pyplot as plt\n", "\n", "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n", "\n", "# Prediction distribution\n", "axes[0, 0].hist(results['predictions']['signal'], bins=3, alpha=0.7)\n", "axes[0, 0].set_title('Signal Distribution')\n", "axes[0, 0].set_xlabel('Signal (-1=sell, 0=abstain, 1=buy)')\n", "\n", "# Confidence over time\n", "axes[0, 1].plot(results['predictions']['confidence'])\n", "axes[0, 1].set_title('Confidence Over Time')\n", "axes[0, 1].set_ylabel('Confidence')\n", "\n", "# Accuracy by confidence\n", "conf_bins = np.linspace(0, 1, 10)\n", "bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2\n", "accuracies = []\n", "for i in range(len(conf_bins)-1):\n", "    mask = (results['predictions']['confidence'] >= conf_bins[i]) & (results['predictions']['confidence'] < conf_bins[i+1])\n", "    if mask.sum() > 0:\n", "        pred_dir = np.where(results['predictions'].loc[mask, 'signal'] > 0, 1, -1)\n", "        actual_dir = results['actuals'].loc[mask, 'actual_direction']\n", "        acc = (pred_dir == actual_dir).mean()\n", "        accuracies.append(acc)\n", "    else:\n", "        accuracies.append(0)\n", "\n", "axes[1, 0].plot(bin_centers, accuracies, 'o-')\n", "axes[1, 0].set_title('Accuracy by Confidence Level')\n", "axes[1, 0].set_xlabel('Confidence')\n", "axes[1, 0].set_ylabel('Accuracy')\n", "\n", "# Returns distribution\n", "axes[1, 1].hist(results['actuals']['actual_return'], bins=50, alpha=0.7)\n", "axes[1, 1].set_title('Actual Returns Distribution')\n", "axes[1, 1].set_xlabel('Return')\n", "\n", "plt.tight_layout()\n", "plt.savefig(RESULTS_DIR / f\"{TICKER}_backtest_plots.png\")\n", "print(f\"✓ Saved plots to {RESULTS_DIR / f'{TICKER}_backtest_plots.png'}\")\n", "plt.show()"]}
    ]
    return cells

def save_notebook(cells, filename):
    """Save notebook to file."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    filepath = NOTEBOOKS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {filename}")

if __name__ == "__main__":
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    
    save_notebook(create_notebook_01(), "01_feature_engineering.ipynb")
    save_notebook(create_notebook_02(), "02_train_price_models.ipynb")
    save_notebook(create_notebook_03(), "03_train_sentiment_models.ipynb")
    save_notebook(create_notebook_04(), "04_train_meta_ensemble.ipynb")
    save_notebook(create_notebook_05(), "05_walkforward_backtest.ipynb")
    
    print("\n✓ All notebooks created!")


