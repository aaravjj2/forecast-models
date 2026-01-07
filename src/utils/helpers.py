"""Helper functions for data processing and validation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def validate_dataframe(df: pd.DataFrame, required_columns: list, 
                      check_nulls: bool = True) -> bool:
    """Validate dataframe has required columns and optionally no nulls."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if check_nulls:
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            print(f"Warning: Null values found:\n{null_counts[null_counts > 0]}")
    
    return True


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fill_value: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros and nulls."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill_value)


def align_time_series(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Remove any future data leakage by filtering to dates <= target_date."""
    if 'date' in df.columns:
        return df[df['date'] <= target_date].copy()
    elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
        return df[df.index <= target_date].copy()
    return df


def calculate_returns(prices: pd.Series, periods: list = [1, 3, 5, 10]) -> pd.DataFrame:
    """Calculate returns over multiple periods."""
    returns = pd.DataFrame(index=prices.index)
    for period in periods:
        returns[f'return_{period}d'] = prices.pct_change(period)
    return returns


def ensure_no_leakage(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                     date_col: str = 'date') -> tuple:
    """Ensure no data leakage between train and test sets."""
    if date_col in train_data.columns:
        max_train_date = train_data[date_col].max()
        min_test_date = test_data[date_col].min()
    else:
        max_train_date = train_data.index.max()
        min_test_date = test_data.index.min()
    
    if max_train_date >= min_test_date:
        raise ValueError(f"Data leakage detected! Train max: {max_train_date}, Test min: {min_test_date}")
    
    return train_data, test_data


def save_artifact(data: Any, filepath: str, format: str = 'pickle'):
    """Save data artifact in specified format."""
    from pathlib import Path
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=True)
        else:
            raise ValueError("CSV format only supports DataFrames")
    elif format == 'json':
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_artifact(filepath: str, format: str = 'pickle'):
    """Load data artifact from specified format."""
    from pathlib import Path
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    
    if format == 'pickle':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        return pd.read_csv(path, index_col=0, parse_dates=True)
    elif format == 'json':
        import json
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


