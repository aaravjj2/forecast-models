"""
Live Signal Generator

Generates trading signals based on trained regime models.
Enforces T+1 Open execution (signal today → execute tomorrow at open).

Signal Types:
    ENTER_LONG: Enter long position
    EXIT_TO_CASH: Exit to cash
    HOLD: Maintain current position
"""

import os
import sys
import logging
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler, MarketRegime, TrendQuality, LiquidityRegime
from strategies.regime_lattice import LatticeState, RegimeLattice
from utils.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Signal = Literal["ENTER_LONG", "EXIT_TO_CASH", "HOLD"]


@dataclass
class SignalResult:
    """Result of signal generation."""
    signal: Signal
    lattice_state: LatticeState
    regime_probs: Dict[str, float]
    confidence: float
    generated_at: datetime
    execute_at: str  # "next_market_open"
    notes: str = ""


class LiveSignalGenerator:
    """
    Generates trading signals from trained regime models.
    
    Usage:
        gen = LiveSignalGenerator(symbol="SPY")
        result = gen.generate_signal()
        print(f"Signal: {result.signal}, State: {result.lattice_state}")
    """
    
    def __init__(
        self,
        symbol: str = "SPY",
        models_dir: Optional[Path] = None,
        lookback_days: int = 252  # 1 year of data for feature calculation
    ):
        """
        Initialize signal generator.
        
        Args:
            symbol: Trading symbol
            models_dir: Directory containing trained regime models
            lookback_days: Days of historical data to fetch for features
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        
        # Model paths
        project_root = Path(__file__).parent.parent.parent
        self.models_dir = models_dir or project_root / "models" / "regimes"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.vol_model = self._load_model("vol_regime_model.pkl")
        self.trend_model = self._load_model("trend_regime_model.pkl")
        self.liq_model = self._load_model("liq_regime_model.pkl")
        
        # Feature builders
        self.price_fetcher = PriceFetcher()
        self.feature_builder = FeatureBuilder()
        self.regime_labeler = RegimeLabeler()
        
        logger.info(f"LiveSignalGenerator initialized for {symbol}")
    
    def _load_model(self, filename: str) -> Optional[Any]:
        """Load a trained model from disk."""
        path = self.models_dir / filename
        if path.exists():
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model: {filename}")
            return model
        else:
            logger.warning(f"Model not found: {path}")
            return None
    
    def save_model(self, model: Any, filename: str):
        """Save a trained model to disk."""
        path = self.models_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model: {filename}")
    
    def _fetch_latest_data(self) -> pd.DataFrame:
        """Fetch latest price data for feature calculation."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        
        prices = self.price_fetcher.fetch(
            self.symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=False  # Always fetch fresh data
        )
        return prices
    
    def _build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build features for the latest data point."""
        features = self.feature_builder.build_all_features(prices)
        return features
    
    def _predict_regimes(self, features: pd.DataFrame) -> Tuple[int, int, int, Dict[str, float]]:
        """
        Predict regime states from features.
        
        Returns:
            (vol_pred, trend_pred, liq_pred, probabilities_dict)
        """
        # Get latest feature row
        latest = features.iloc[[-1]].drop(columns=['target_return_1d'], errors='ignore')
        
        # Fill any NaNs
        latest = latest.fillna(0)
        
        probs = {}
        
        # Volatility prediction
        if self.vol_model is not None:
            vol_pred = int(self.vol_model.predict(latest)[0])
            try:
                vol_prob = self.vol_model.predict_proba(latest)[0]
                probs['vol_high_prob'] = float(vol_prob[1]) if len(vol_prob) > 1 else float(vol_pred)
            except:
                probs['vol_high_prob'] = float(vol_pred)
        else:
            vol_pred = 0  # Default: Risk On
            probs['vol_high_prob'] = 0.0
        
        # Trend prediction
        if self.trend_model is not None:
            trend_pred = int(self.trend_model.predict(latest)[0])
            try:
                trend_prob = self.trend_model.predict_proba(latest)[0]
                probs['trend_robust_prob'] = float(trend_prob[1]) if len(trend_prob) > 1 else float(trend_pred)
            except:
                probs['trend_robust_prob'] = float(trend_pred)
        else:
            trend_pred = 1  # Default: Robust Trend
            probs['trend_robust_prob'] = 1.0
        
        # Liquidity prediction
        if self.liq_model is not None:
            liq_pred = int(self.liq_model.predict(latest)[0])
            try:
                liq_prob = self.liq_model.predict_proba(latest)[0]
                probs['liq_stressed_prob'] = float(liq_prob[1]) if len(liq_prob) > 1 else float(liq_pred)
            except:
                probs['liq_stressed_prob'] = float(liq_pred)
        else:
            liq_pred = 0  # Default: Normal liquidity
            probs['liq_stressed_prob'] = 0.0
        
        return vol_pred, trend_pred, liq_pred, probs
    
    def generate_signal(self, current_exposure: Literal["LONG", "FLAT"] = "FLAT") -> SignalResult:
        """
        Generate trading signal for next market open.
        
        Args:
            current_exposure: Current portfolio state
            
        Returns:
            SignalResult with signal and metadata
        """
        generated_at = datetime.now(timezone.utc)
        
        # Fetch and process data
        try:
            prices = self._fetch_latest_data()
            features = self._build_features(prices)
        except Exception as e:
            logger.error(f"Data fetch/build failed: {e}")
            return SignalResult(
                signal="HOLD",
                lattice_state=LatticeState.HOSTILE,
                regime_probs={},
                confidence=0.0,
                generated_at=generated_at,
                execute_at="next_market_open",
                notes=f"ERROR: {str(e)}"
            )
        
        # Predict regimes
        vol_pred, trend_pred, liq_pred, probs = self._predict_regimes(features)
        
        # Determine lattice state using RegimeLattice class
        lattice = RegimeLattice()
        # For single-point predictions, use simple logic
        if liq_pred == 1 or vol_pred == 1:
            lattice_state = LatticeState.HOSTILE
        elif trend_pred == 1:
            lattice_state = LatticeState.FAVORABLE
        else:
            lattice_state = LatticeState.NEUTRAL
        
        # Calculate confidence (simple average of prediction confidences)
        confidence = np.mean([
            abs(probs.get('vol_high_prob', 0.5) - 0.5) * 2,
            abs(probs.get('trend_robust_prob', 0.5) - 0.5) * 2,
            abs(probs.get('liq_stressed_prob', 0.5) - 0.5) * 2
        ])
        
        # Generate signal based on lattice state
        if lattice_state == LatticeState.HOSTILE:
            signal = "EXIT_TO_CASH"
        elif lattice_state in [LatticeState.FAVORABLE, LatticeState.NEUTRAL]:
            signal = "ENTER_LONG" if current_exposure == "FLAT" else "HOLD"
        else:
            signal = "HOLD"
        
        return SignalResult(
            signal=signal,
            lattice_state=lattice_state,
            regime_probs=probs,
            confidence=confidence,
            generated_at=generated_at,
            execute_at="next_market_open",
            notes=f"Vol={vol_pred}, Trend={trend_pred}, Liq={liq_pred}"
        )
    
    def train_models(self, prices: Optional[pd.DataFrame] = None):
        """
        Train regime models on historical data.
        
        Should be called once during setup, then models are loaded.
        """
        import lightgbm as lgb
        
        if prices is None:
            prices = self._fetch_latest_data()
        
        features = self._build_features(prices)
        labels = self.regime_labeler.label_regimes(prices)
        
        # Align
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx].drop(columns=['target_return_1d'], errors='ignore').fillna(0)
        
        # Train Volatility model
        y_vol = labels.loc[common_idx, 'regime_vol']
        vol_model = lgb.LGBMClassifier(n_estimators=100, random_state=CONFIG.SEED, verbose=-1)
        vol_model.fit(X, y_vol)
        self.vol_model = vol_model
        self.save_model(vol_model, "vol_regime_model.pkl")
        
        # Train Trend model
        y_trend = labels.loc[common_idx, 'regime_trend']
        trend_model = lgb.LGBMClassifier(n_estimators=100, random_state=CONFIG.SEED, verbose=-1)
        trend_model.fit(X, y_trend)
        self.trend_model = trend_model
        self.save_model(trend_model, "trend_regime_model.pkl")
        
        # Train Liquidity model
        y_liq = labels.loc[common_idx, 'regime_liq']
        liq_model = lgb.LGBMClassifier(n_estimators=100, random_state=CONFIG.SEED, verbose=-1)
        liq_model.fit(X, y_liq)
        self.liq_model = liq_model
        self.save_model(liq_model, "liq_regime_model.pkl")
        
        logger.info("All regime models trained and saved.")


if __name__ == "__main__":
    # Quick test
    gen = LiveSignalGenerator(symbol="SPY")
    
    # Check if models exist, else train
    if gen.vol_model is None:
        print("Training models...")
        gen.train_models()
    
    # Generate signal
    result = gen.generate_signal(current_exposure="FLAT")
    print(f"Signal: {result.signal}")
    print(f"Lattice State: {result.lattice_state}")
    print(f"Probabilities: {result.regime_probs}")
    print(f"Confidence: {result.confidence:.2%}")
