"""LightGBM specialist model for diversity."""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import joblib
from typing import Dict, Any

from .base_model import BaseModel, ModelSignal


class LightGBMModel(BaseModel):
    """LightGBM model for price-based predictions (diversity model)."""
    
    def __init__(self, min_confidence: float = 0.6, **lgb_params):
        super().__init__("LightGBM", min_confidence)
        
        # Default LightGBM parameters
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            **lgb_params
        }
        
        self.model = None
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """Train LightGBM model."""
        # Convert y to binary (0/1)
        y_binary = (y > 0).astype(int)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = ((train_pred > 0.5) == y_train).mean()
        val_acc = ((val_pred > 0.5) == y_val).mean()
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'n_features': len(self.feature_columns),
            'n_estimators': self.model.num_trees()
        }
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        """Get prediction for a single snapshot."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X = pd.DataFrame([snapshot[self.feature_columns]])
        
        prob_up = float(self.model.predict(X)[0])
        confidence = self._calculate_confidence(prob_up)
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'prob_up': prob_up}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict on a batch."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X_aligned = X[self.feature_columns]
        
        prob_up = self.model.predict(X_aligned)
        confidence = np.array([self._calculate_confidence(p) for p in prob_up])
        signal = np.array([self._apply_abstention(p, c) 
                          for p, c in zip(prob_up, confidence)])
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        """Save model and metadata."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'params': self.params,
            'min_confidence': self.min_confidence
        }
        joblib.dump(metadata, str(path).replace('.txt', '_metadata.pkl'))
    
    def load(self, filepath: str):
        """Load model and metadata."""
        path = Path(filepath)
        
        # Load model
        self.model = lgb.Booster(model_file=str(path))
        
        # Load metadata
        metadata = joblib.load(str(path).replace('.txt', '_metadata.pkl'))
        self.feature_columns = metadata['feature_columns']
        self.params = metadata.get('params', self.params)
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True



