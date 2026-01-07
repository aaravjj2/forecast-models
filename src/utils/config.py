"""Configuration management for the pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / "keys.env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Try loading from current directory or parent
    load_dotenv()

# API Keys - Load from environment (set from Colab secrets or keys.env)
# All keys match the names in keys.env
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB2_API_KEY = os.getenv("FINNHUB2_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
SEC_API_KEY = os.getenv("SEC_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
FINAGE_API_KEY = os.getenv("FINAGE_API_KEY", "")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SPECIALIST_MODELS_DIR = MODELS_DIR / "specialists"
META_MODEL_DIR = MODELS_DIR / "meta"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPECIALIST_MODELS_DIR, 
                 META_MODEL_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
DEFAULT_TRAIN_WINDOW_DAYS = 252  # 1 year
DEFAULT_TEST_WINDOW_DAYS = 21    # 1 month
DEFAULT_MIN_CONFIDENCE = 0.6     # Abstention threshold

# Feature engineering
LOOKBACK_WINDOWS = [1, 3, 5, 10, 20, 50]  # Days for returns/volatility
MA_WINDOWS = [5, 10, 20, 50, 200]  # Moving average windows

# Backtesting
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001  # 0.1%

# --- HARDENED EXPERIMENT CONFIG (Phase 1-3) ---
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Reproducibility
    SEED: int = 42
    
    # Data
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2024-01-01"
    
    # Training (Fixed Rolling Window)
    TRAIN_WINDOW_DAYS: int = 500
    PREDICT_WINDOW_DAYS: int = 60 # Retrain every 2 months
    
    # Feature Engineering
    FORECAST_HORIZON: int = 5 # For Regime Labeling
    
    # Execution (Reality)
    COST_BPS: float = 10.0
    STRESS_COST_BPS: float = 15.0
    SLIPPAGE_STD: float = 0.0005 # 5 bps random slippage
    
    # Regime Logic
    # Thresholds for RegimeLabeler
    TREND_ADX_THRESHOLD: float = 25.0
    VOL_PERCENTILE: float = 0.75 # Top 25% Vol is High Vol
    LIQ_PERCENTILE: float = 0.85 # Top 15% Illiquidity is Stressed
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

CONFIG = ExperimentConfig()

