"""XGBoost specialist model for price pattern learning."""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
from typing import Dict, Any

from .base_model import BaseModel, ModelSignal


class XGBoostModel(BaseModel):
    """XGBoost model for price-based predictions."""
    
    def __init__(self, min_confidence: float = 0.6, **xgb_params):
        super().__init__("XGBoost", min_confidence)
        
        # Default XGBoost parameters (free-tier friendly)
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',  # Faster, works on CPU
            'base_score': 0.5,  # Fix for logistic loss
            **xgb_params
        }
        
        self.model = None
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """Train XGBoost model."""
        # Convert y to binary (0/1) for XGBoost
        y_binary = (y > 0).astype(int)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        train_acc = ((train_pred > 0.5) == y_train).mean()
        val_acc = ((val_pred > 0.5) == y_val).mean()
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'n_features': len(self.feature_columns),
            'n_estimators': self.model.num_boosted_rounds()
        }
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        """Get prediction for a single snapshot."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure correct feature order
        X = pd.DataFrame([snapshot[self.feature_columns]])
        dtest = xgb.DMatrix(X)
        
        prob_up = float(self.model.predict(dtest)[0])
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
        dtest = xgb.DMatrix(X_aligned)
        
        prob_up = self.model.predict(dtest)
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
        joblib.dump(metadata, str(path).replace('.model', '_metadata.pkl'))
    
    def load(self, filepath: str):
        """Load model and metadata."""
        path = Path(filepath)
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        
        # Load metadata
        metadata = joblib.load(str(path).replace('.model', '_metadata.pkl'))
        self.feature_columns = metadata['feature_columns']
        self.params = metadata.get('params', self.params)
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True

