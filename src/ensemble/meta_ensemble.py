"""Meta-gating ensemble that combines specialist models."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import joblib

# Robust imports
import sys
from pathlib import Path
try:
    from ..models.base_model import BaseModel, ModelSignal
    from ..utils.config import DEFAULT_MIN_CONFIDENCE
except ImportError:
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from models.base_model import BaseModel, ModelSignal
    from utils.config import DEFAULT_MIN_CONFIDENCE


class MetaEnsemble:
    """Meta-model that learns which specialist to trust and when."""
    
    def __init__(self, specialists: List[BaseModel], min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                 retrain_specialists: bool = False):
        self.specialists = specialists
        self.min_confidence = min_confidence
        self.retrain_specialists = retrain_specialists
        self.meta_model = None
        self.is_trained = False
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              market_features: Optional[pd.DataFrame] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train meta-model to combine specialist predictions.
        
        Args:
            X: Feature matrix
            y: Target variable
            market_features: Optional market regime features (volatility, etc.)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Get predictions from all specialists
        specialist_preds = {}
        specialist_preds = {}
        for specialist in self.specialists:
            if self.retrain_specialists:
                print(f"Retraining {specialist.name} on current window...")
                try:
                    # Retrain on current window
                    specialist.train(X, y)
                except Exception as e:
                    print(f"Error retraining {specialist.name}: {e}")
                    continue
            
            if not specialist.is_trained:
                print(f"Warning: {specialist.name} not trained, skipping")
                continue
            
            try:
                preds = specialist.predict_batch(X)
                specialist_preds[specialist.name] = preds
            except Exception as e:
                print(f"Error getting predictions from {specialist.name}: {e}")
                continue
        
        if not specialist_preds:
            raise ValueError("No trained specialists available")
        
        # Build meta-features
        meta_features = []
        feature_names = []
        
        # Specialist outputs
        for name, preds in specialist_preds.items():
            meta_features.append(preds['prob_up'].values)
            meta_features.append(preds['confidence'].values)
            feature_names.extend([f'{name}_prob', f'{name}_conf'])
        
        # Market regime features
        if market_features is not None:
            # Add volatility regime
            if 'volatility' in market_features.columns:
                meta_features.append(market_features['volatility'].values)
                feature_names.append('market_volatility')
            
            # Add news intensity
            news_cols = [col for col in market_features.columns if 'news' in col.lower()]
            if news_cols:
                meta_features.append(market_features[news_cols[0]].values)
                feature_names.append('news_intensity')
        
        # Combine meta-features
        meta_X = np.column_stack(meta_features)
        self.feature_columns = feature_names
        
        # Target: did the ensemble prediction match actual direction?
        y_binary = (y > 0).astype(int)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            meta_X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train meta-model (Random Forest for interpretability)
        self.meta_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.meta_model.predict(X_train)
        val_pred = self.meta_model.predict(X_val)
        
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()
        
        self.is_trained = True
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'n_specialists': len(specialist_preds),
            'n_meta_features': len(feature_names)
        }
    
    def get_signal(self, snapshot: pd.Series, 
                   market_snapshot: Optional[pd.Series] = None) -> ModelSignal:
        """
        Get ensemble prediction for a single snapshot.
        
        Args:
            snapshot: Feature snapshot
            market_snapshot: Optional market regime features
        
        Returns:
            Combined ModelSignal with abstention
        """
        if not self.is_trained:
            raise ValueError("Meta-ensemble not trained yet")
        
        # Get predictions from all specialists
        specialist_signals = []
        for specialist in self.specialists:
            if not specialist.is_trained:
                continue
            
            try:
                signal = specialist.get_signal(snapshot)
                specialist_signals.append((specialist.name, signal))
            except Exception as e:
                print(f"Error from {specialist.name}: {e}")
                continue
        
        if not specialist_signals:
            # Fallback: return abstention
            return ModelSignal(signal=0, prob_up=0.5, confidence=0.0)
        
        # Build meta-features
        meta_features = []
        for name, signal in specialist_signals:
            meta_features.append(signal.prob_up)
            meta_features.append(signal.confidence)
        
        # Add market features if available
        if market_snapshot is not None:
            if 'volatility' in market_snapshot.index:
                meta_features.append(market_snapshot['volatility'])
            news_cols = [col for col in market_snapshot.index if 'news' in col.lower()]
            if news_cols:
                meta_features.append(market_snapshot[news_cols[0]])
        
        # Get meta-model prediction
        meta_X = np.array([meta_features])
        prob_up = self.meta_model.predict_proba(meta_X)[0][1]
        
        # Calculate confidence (weighted average of specialist confidences)
        confidences = [s.confidence for _, s in specialist_signals]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Apply abstention
        if avg_confidence < self.min_confidence:
            signal = 0  # Abstain
        elif prob_up > 0.5:
            signal = 1  # Buy
        else:
            signal = -1  # Sell
        
        return ModelSignal(
            signal=signal,
            prob_up=float(prob_up),
            confidence=float(avg_confidence),
            raw_output={
                'specialist_signals': {name: s.signal for name, s in specialist_signals},
                'meta_prob': prob_up
            }
        )
    
    def predict_batch(self, X: pd.DataFrame,
                     market_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Predict on a batch."""
        if not self.is_trained:
            raise ValueError("Meta-ensemble not trained yet")
        
        # Get predictions from all specialists
        specialist_preds = {}
        for specialist in self.specialists:
            if not specialist.is_trained:
                continue
            
            try:
                preds = specialist.predict_batch(X)
                specialist_preds[specialist.name] = preds
            except Exception as e:
                print(f"Error from {specialist.name}: {e}")
                continue
        
        if not specialist_preds:
            # Return abstention for all
            return pd.DataFrame({
                'signal': 0,
                'prob_up': 0.5,
                'confidence': 0.0
            }, index=X.index)
        
        # Build meta-features
        meta_features_list = []
        for name, preds in specialist_preds.items():
            meta_features_list.append(preds['prob_up'].values)
            meta_features_list.append(preds['confidence'].values)
        
        # Add market features if available
        if market_features is not None:
            if 'volatility' in market_features.columns:
                meta_features_list.append(market_features['volatility'].values)
            news_cols = [col for col in market_features.columns if 'news' in col.lower()]
            if news_cols:
                meta_features_list.append(market_features[news_cols[0]].values)
        
        # Get meta-model predictions
        meta_X = np.column_stack(meta_features_list)
        prob_up = self.meta_model.predict_proba(meta_X)[:, 1]
        
        # Calculate confidence (average of specialist confidences)
        confidences = [preds['confidence'].values for preds in specialist_preds.values()]
        avg_confidence = np.mean(confidences, axis=0)
        
        # Apply abstention
        signal = np.where(avg_confidence < self.min_confidence, 0,
                         np.where(prob_up > 0.5, 1, -1))
        
        return pd.DataFrame({
            'signal': signal,
            'prob_up': prob_up,
            'confidence': avg_confidence
        }, index=X.index)
    
    def save(self, filepath: str):
        """Save meta-ensemble."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_columns': self.feature_columns,
            'min_confidence': self.min_confidence,
            'specialist_names': [s.name for s in self.specialists]
        }
        
        # Save meta-model
        joblib.dump(self.meta_model, str(path))
        
        # Save metadata
        joblib.dump(metadata, str(path).replace('.pkl', '_metadata.pkl'))
    
    def load(self, filepath: str):
        """Load meta-ensemble."""
        path = Path(filepath)
        
        # Load meta-model
        self.meta_model = joblib.load(str(path))
        
        # Load metadata
        metadata = joblib.load(str(path).replace('.pkl', '_metadata.pkl'))
        self.feature_columns = metadata['feature_columns']
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        
        self.is_trained = True

