"""Base class for all prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class ModelSignal:
    """Standardized model output."""
    signal: int  # -1 (sell), 0 (abstain), 1 (buy)
    prob_up: float  # Probability of positive return [0, 1]
    confidence: float  # Model confidence [0, 1]
    raw_output: Optional[Any] = None  # Raw model output for debugging


class BaseModel(ABC):
    """Base class that all specialist models must implement."""
    
    def __init__(self, name: str, min_confidence: float = 0.6):
        self.name = name
        self.min_confidence = min_confidence
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable (binary: -1, 0, 1)
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        """
        Get prediction signal for a single snapshot.
        
        Args:
            snapshot: Single row of features (pd.Series)
        
        Returns:
            ModelSignal with prediction
        """
        pass
    
    @abstractmethod
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a batch of samples.
        
        Args:
            X: Feature matrix
        
        Returns:
            DataFrame with columns: signal, prob_up, confidence
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model from disk."""
        pass
    
    def _apply_abstention(self, prob_up: float, confidence: float) -> int:
        """
        Apply abstention logic based on confidence threshold.
        
        Returns:
            signal: -1, 0, or 1 (0 = abstain)
        """
        if confidence < self.min_confidence:
            return 0  # Abstain
        
        if prob_up > 0.5:
            return 1  # Buy
        elif prob_up < 0.5:
            return -1  # Sell
        else:
            return 0  # Abstain (uncertain)
    
    def _calculate_confidence(self, prob_up: float) -> float:
        """
        Calculate confidence from probability.
        Higher confidence when probability is further from 0.5.
        """
        return 2 * abs(prob_up - 0.5)  # Maps [0.5, 1.0] -> [0, 1]


