"""
Regime Lattice Logic - Phase C
Combines orthogonal regime predictions into a single Master State.
"""

from enum import IntEnum, auto
import pandas as pd
import numpy as np

class LatticeState(IntEnum):
    HOSTILE = 0     # Flat (Risk Off or Illiquid)
    NEUTRAL = 1     # Mean Reversion (Risk On + Chop)
    FAVORABLE = 2   # Momentum (Risk On + Trend)

class RegimeLattice:
    """
    Combines independent regime predictions into a single actionable market state.
    3D Lattice (Minimal):
    1. Volatility (0=RiskOn, 1=RiskOff)
    2. Trend Quality (0=Noisy, 1=Robust)
    3. Liquidity (0=Normal, 1=Stressed)
    """
    
    def __init__(self):
        pass
        
    def determine_state(self, 
                       pred_vol: pd.Series, 
                       pred_trend: pd.Series,
                       pred_liq: pd.Series) -> pd.Series:
        """
        Determine Lattice State from component predictions.
        """
        
        # Align inputs (intersection)
        common_idx = pred_vol.index.intersection(pred_trend.index).intersection(pred_liq.index)
        
        vol = pred_vol.loc[common_idx]
        trend = pred_trend.loc[common_idx]
        liq = pred_liq.loc[common_idx]
        
        states = pd.Series(LatticeState.NEUTRAL, index=common_idx)
        
        # 1. HOSTILE (Safety First)
        # If Vol is Risk Off (1) OR Liquidity is Stressed (1) -> HOSTILE
        mask_hostile = (vol == 1) | (liq == 1)
        states[mask_hostile] = LatticeState.HOSTILE
        
        # 2. FAVORABLE (Efficiency)
        # If Safe (Vol=0, Liq=0) AND Trend represents Opportunity (1)
        mask_safe = (~mask_hostile)
        
        mask_favorable = mask_safe & (trend == 1)
        
        # 3. NEUTRAL (Chop)
        # If Safe (Vol=0, Liq=0) AND Trend is Noisy (0) -> NEUTRAL
        mask_neutral = mask_safe & (trend == 0)
        
        states[mask_favorable] = LatticeState.FAVORABLE
        states[mask_neutral] = LatticeState.NEUTRAL
        
        return states

    def describe_state(self, state: int) -> str:
        return LatticeState(state).name
