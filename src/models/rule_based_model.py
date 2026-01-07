"""Simple rule-based baseline model."""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any

from .base_model import BaseModel, ModelSignal


class RuleBasedModel(BaseModel):
    """Simple rule-based model for baseline comparison."""
    
    def __init__(self, min_confidence: float = 0.6):
        super().__init__("RuleBased", min_confidence)
        self.rules = {}
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train rule-based model (learn simple thresholds)."""
        self.feature_columns = X.columns.tolist()
        
        # Learn simple rules based on feature correlations
        # Rule 1: Momentum-based
        momentum_cols = [col for col in X.columns if 'momentum' in col.lower() or 'return' in col.lower()]
        if momentum_cols:
            best_momentum = max(momentum_cols, key=lambda col: abs(X[col].corr(y)))
            self.rules['momentum_col'] = best_momentum
            self.rules['momentum_threshold'] = X[best_momentum].quantile(0.6)
        
        # Rule 2: RSI-based
        rsi_cols = [col for col in X.columns if 'rsi' in col.lower()]
        if rsi_cols:
            best_rsi = max(rsi_cols, key=lambda col: abs(X[col].corr(y)))
            self.rules['rsi_col'] = best_rsi
            self.rules['rsi_oversold'] = X[best_rsi].quantile(0.2)  # Buy when oversold
            self.rules['rsi_overbought'] = X[best_rsi].quantile(0.8)  # Sell when overbought
        
        # Rule 3: Moving average crossover
        ma_cols = [col for col in X.columns if 'ma_' in col.lower() and '_ratio' in col.lower()]
        if ma_cols:
            best_ma = max(ma_cols, key=lambda col: abs(X[col].corr(y)))
            self.rules['ma_col'] = best_ma
            self.rules['ma_threshold'] = 1.0  # Above MA = bullish
        
        self.is_trained = True
        
        # Calculate baseline accuracy
        predictions = self._apply_rules(X)
        accuracy = (predictions == (y > 0).astype(int)).mean()
        
        return {
            'train_accuracy': float(accuracy),
            'n_rules': len(self.rules),
            'n_features': len(self.feature_columns)
        }
    
    def _apply_rules(self, X: pd.DataFrame) -> np.ndarray:
        """Apply learned rules to get predictions."""
        predictions = np.zeros(len(X))
        
        # Rule 1: Momentum
        if 'momentum_col' in self.rules:
            col = self.rules['momentum_col']
            threshold = self.rules['momentum_threshold']
            if col in X.columns:
                predictions += (X[col] > threshold).astype(int) * 0.4
        
        # Rule 2: RSI
        if 'rsi_col' in self.rules:
            col = self.rules['rsi_col']
            oversold = self.rules.get('rsi_oversold', 30)
            overbought = self.rules.get('rsi_overbought', 70)
            if col in X.columns:
                # Buy when oversold, sell when overbought
                buy_signal = (X[col] < oversold).astype(int) * 0.3
                sell_signal = (X[col] > overbought).astype(int) * -0.3
                predictions += buy_signal + sell_signal
        
        # Rule 3: Moving average
        if 'ma_col' in self.rules:
            col = self.rules['ma_col']
            threshold = self.rules.get('ma_threshold', 1.0)
            if col in X.columns:
                predictions += (X[col] > threshold).astype(int) * 0.3
        
        # Convert to binary
        return (predictions > 0).astype(int)
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        """Get prediction from rules."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Apply rules
        X = pd.DataFrame([snapshot[self.feature_columns]])
        prediction = self._apply_rules(X)[0]
        
        # Convert to probability
        prob_up = float(prediction)
        confidence = 0.5  # Rule-based models have lower confidence
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'rule_prediction': prediction}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict on a batch."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self._apply_rules(X)
        prob_up = predictions.astype(float)
        confidence = np.full(len(X), 0.5)  # Low confidence for rules
        signal = np.array([self._apply_abstention(p, c) 
                          for p, c in zip(prob_up, confidence)])
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        """Save model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_columns': self.feature_columns,
            'rules': self.rules,
            'min_confidence': self.min_confidence
        }
        joblib.dump(metadata, str(path))
    
    def load(self, filepath: str):
        """Load model."""
        path = Path(filepath)
        metadata = joblib.load(str(path))
        
        self.feature_columns = metadata['feature_columns']
        self.rules = metadata['rules']
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True



