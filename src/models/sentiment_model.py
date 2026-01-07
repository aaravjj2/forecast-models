"""Sentiment-based prediction model using FinBERT/FinGPT."""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any, Optional

from .base_model import BaseModel, ModelSignal


class SentimentModel(BaseModel):
    """Sentiment model using news data and pretrained transformers."""
    
    def __init__(self, min_confidence: float = 0.6, use_pretrained: bool = True):
        super().__init__("Sentiment", min_confidence)
        self.use_pretrained = use_pretrained
        self.sentiment_model = None
        self.tokenizer = None
        self.feature_columns = None
        
        # Initialize transformer if requested
        if use_pretrained:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                # Use FinBERT (free, works on CPU)
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.sentiment_model.eval()  # Set to eval mode
                
                # Use CPU for free tier
                self.device = torch.device('cpu')
                self.sentiment_model.to(self.device)
                
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"Warning: Could not load FinBERT: {e}")
                print("Falling back to simple sentiment")
                self.use_pretrained = False
    
    def _get_sentiment_score(self, text: str) -> float:
        """Get sentiment score for text."""
        if pd.isna(text) or not text:
            return 0.0
        
        if self.use_pretrained and self.sentiment_model is not None:
            return self._get_finbert_sentiment(text)
        else:
            return self._get_simple_sentiment(text)
    
    def _get_finbert_sentiment(self, text: str) -> float:
        """Get sentiment using FinBERT."""
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, max_length=512,
                                  padding=True).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: positive, negative, neutral
            # Map to [-1, 1] scale
            positive_prob = probs[0][0].item()
            negative_prob = probs[0][1].item()
            neutral_prob = probs[0][2].item()
            
            # Return sentiment score: positive - negative
            sentiment = positive_prob - negative_prob
            return sentiment
            
        except Exception as e:
            print(f"Error in FinBERT prediction: {e}")
            return self._get_simple_sentiment(text)
    
    def _get_simple_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment fallback."""
        text_lower = str(text).lower()
        
        positive_words = ['up', 'gain', 'rise', 'surge', 'rally', 'bullish', 
                         'growth', 'profit', 'beat', 'strong', 'positive', 'buy']
        negative_words = ['down', 'fall', 'drop', 'decline', 'crash', 'bearish',
                         'loss', 'miss', 'weak', 'negative', 'warn', 'sell']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count + 1)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              news_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Train sentiment model.
        
        Args:
            X: Feature matrix (should include news features)
            y: Target variable
            news_data: Optional raw news DataFrame for training
        """
        # For sentiment model, we primarily use news features
        # Store which columns are news-related
        news_cols = [col for col in X.columns if 'news' in col.lower() or 'sentiment' in col.lower()]
        self.feature_columns = news_cols if news_cols else X.columns.tolist()
        
        # Simple training: learn weights for news features
        # In practice, this could be more sophisticated
        if len(news_cols) > 0:
            # Use news sentiment features directly
            X_sentiment = X[news_cols].fillna(0)
            
            # Simple linear combination
            # Weight by correlation with target
            correlations = X_sentiment.apply(lambda col: col.corr(y))
            self.feature_weights = correlations.fillna(0).to_dict()
        else:
            # Fallback: use all features
            self.feature_weights = {col: 1.0 / len(X.columns) for col in X.columns}
        
        self.is_trained = True
        
        # Calculate baseline accuracy
        if len(news_cols) > 0:
            X_sentiment = X[news_cols].fillna(0)
            weighted_sentiment = sum(X_sentiment[col] * self.feature_weights.get(col, 0) 
                                    for col in news_cols)
            predictions = (weighted_sentiment > 0).astype(int)
            accuracy = (predictions == (y > 0).astype(int)).mean()
        else:
            accuracy = 0.5  # Random baseline
        
        return {
            'train_accuracy': float(accuracy),
            'n_news_features': len(news_cols),
            'use_pretrained': self.use_pretrained
        }
    
    def get_signal(self, snapshot: pd.Series) -> ModelSignal:
        """Get prediction from sentiment features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get sentiment score from news features
        sentiment_score = 0.0
        
        for col, weight in self.feature_weights.items():
            if col in snapshot:
                sentiment_score += snapshot[col] * weight
        
        # Convert sentiment score [-1, 1] to prob_up [0, 1]
        prob_up = (sentiment_score + 1) / 2
        prob_up = np.clip(prob_up, 0.0, 1.0)
        
        confidence = self._calculate_confidence(prob_up)
        signal = self._apply_abstention(prob_up, confidence)
        
        return ModelSignal(
            signal=signal,
            prob_up=prob_up,
            confidence=confidence,
            raw_output={'sentiment_score': sentiment_score}
        )
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict on a batch."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Calculate weighted sentiment
        sentiment_scores = np.zeros(len(X))
        
        for col, weight in self.feature_weights.items():
            if col in X.columns:
                sentiment_scores += X[col].fillna(0) * weight
        
        # Convert to prob_up
        prob_up = (sentiment_scores + 1) / 2
        prob_up = np.clip(prob_up, 0.0, 1.0)
        
        confidence = np.array([self._calculate_confidence(p) for p in prob_up])
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
            'feature_weights': self.feature_weights,
            'min_confidence': self.min_confidence,
            'use_pretrained': self.use_pretrained
        }
        joblib.dump(metadata, str(path))
    
    def load(self, filepath: str):
        """Load model."""
        path = Path(filepath)
        metadata = joblib.load(str(path))
        
        self.feature_columns = metadata['feature_columns']
        self.feature_weights = metadata['feature_weights']
        self.min_confidence = metadata.get('min_confidence', self.min_confidence)
        self.use_pretrained = metadata.get('use_pretrained', False)
        
        self.is_trained = True



