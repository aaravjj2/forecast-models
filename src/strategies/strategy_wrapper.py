"""
Strategy Wrapper - Phase D1
Maps predicted regimes to fixed, simple strategies without optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.strategies.regime_lattice import LatticeState

class StrategyWrapper:
    """
    Executes simple strategies based on the predicted regime.
    Strategies are FIXED rules (no ML).
    """
    
    def __init__(self, regime_col: str = 'predicted_regime'):
        self.regime_col = regime_col
        
    def generate_signals(self, prices: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals scaled by Regime Lattice State.
        
        Lattice Logic:
        - HOSTILE (0): Flat (Signal = 0)
        - NEUTRAL (1): Mean Reversion (Signal = +/- 0.5 Size)
        - FAVORABLE (2): Momentum (Signal = +/- 1.0 Size)
        """
        df = prices.copy()
        signals = pd.Series(0.0, index=df.index)
        
        # Calculate Technicals necessary for strategies
        df['ma_50'] = df['close'].rolling(50).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Align regimes
        regime_series = regimes.reindex(df.index).fillna(LatticeState.NEUTRAL)
        
        # Strategy 1: Momentum (Active in Favorable)
        mom_signal = np.where(df['close'] > df['ma_50'], 1.0, -1.0)
        
        # Strategy 2: Mean Reversion (Active in Neutral)
        # Reduced size (0.5)
        mr_signal = np.where(df['rsi_14'] < 30, 0.5, 
                            np.where(df['rsi_14'] > 70, -0.5, 0.0))
        
        # Apply Lattice Logic
        
        # Mask for FAVORABLE (Risk-On + Trend -> Long)
        mask_fav = (regime_series == LatticeState.FAVORABLE)
        signals[mask_fav] = 1.0
        
        # Mask for NEUTRAL (Risk-On + Chop -> Long)
        mask_neu = (regime_series == LatticeState.NEUTRAL)
        signals[mask_neu] = 1.0
        
        # Mask for HOSTILE (Risk-Off or Illiquid -> Flat)
        mask_hostile = (regime_series == LatticeState.HOSTILE)
        signals[mask_hostile] = 0.0
        
        # Compatibility with legacy MarketRegime inputs (for Benchmarks)
        # If input is MarketRegime (not LatticeState), map manually:
        # TRENDING -> Favorable (Mom)
        # LOW_VOL/NEUTRAL -> Neutral (MR)
        # HIGH_VOL/CRASH -> Hostile (Flat)
        # Check first element type
        if len(regimes) > 0 and not isinstance(regimes.iloc[0], (int, np.integer, LatticeState)):
             # Assume it might be mapped already, or fallback
             pass
        elif len(regimes) > 0 and regimes.iloc[0] in [1, 2, 0]: # IntEnum values overlap... 
             # MarketRegime.TRENDING = 1, LatticeState.NEUTRAL = 1.
             # This is tricky.
             # We should rely on the caller to pass LatticeState objects or correct Ints.
             pass
        
        return signals

    def backtest_conditional(self, prices: pd.DataFrame, signals: pd.Series, cost_bps: float = 10.0) -> Dict[str, float]:
        """
        Simple vectorized backtest of the signals.
        Returns basic metrics.
        cost_bps: Basis points per trade side (Turnover). 
                  Default 10bps = 0.0010 impact on price per 1.0 position change.
        """
        # Align
        common_idx = prices.index.intersection(signals.index)
        px = prices.loc[common_idx]
        sig = signals.loc[common_idx]
        
        # Shift signal by 1 (Trade at Open of T+1 based on Signal at Close T)
        # Returns: Open to Open t+1 to t+2?
        # Decision at Close T. Execution at Open T+1.
        
        next_day_ret = (px['open'].shift(-2) - px['open'].shift(-1)) / px['open'].shift(-1)
        
        # Gross Return
        gross_ret = sig * next_day_ret
        
        # Costs
        # Turnover = Abs(Signal(t) - Signal(t-1))
        # Calculated at T (Executed at T+1).
        # Cost is incurred on the Open of T+1.
        turnover = sig.diff().abs().fillna(0)
        # Initial entry cost (index 0)
        if len(sig) > 0 and sig.iloc[0] != 0:
            turnover.iloc[0] = abs(sig.iloc[0])
            
        cost_pct = turnover * (cost_bps / 10000.0)
        
        # Net Return = Gross - Cost
        # Note: Cost reduces capital. Approximation: Ret = Ret - Cost.
        net_ret = gross_ret - cost_pct
        
        net_ret = net_ret.dropna()
        
        total_return = (1 + net_ret).prod() - 1
        sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(252) if net_ret.std() > 0 else 0
        
        # Win Rate
        wins = (net_ret > 0).sum()
        total = (net_ret != 0).count()
        win_rate = wins / total if total > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'n_trades': total, # Days Active
            'turnover_count': turnover.sum() # Total Units Traded
        }
