"""
Regime Labeler - Phase 2 (Rebuild)
Generates Ground Truth labels for the 3-Regime Lattice:
1. Volatility (Primary)
2. Trend Quality (Secondary)
3. Liquidity (Safety)
"""

import pandas as pd
import numpy as np
from enum import IntEnum
import sys
from pathlib import Path

# Import CONFIG
try:
    from ..utils.config import CONFIG
except ImportError:
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import CONFIG

class MarketRegime(IntEnum):
    NEUTRAL = 0     # Risk On
    HIGH_VOL = 1    # Risk Off (includes Crash)

class TrendQuality(IntEnum):
    NOISY = 0       # Chop / Low ADX
    ROBUST = 1      # Trending / High ADX

class LiquidityRegime(IntEnum):
    NORMAL = 0
    STRESSED = 1    # Kill Switch

class RegimeLabeler:
    """
    Labels historical data with Ground Truth regimes.
    Strict separation of Future Data (Labels) vs Past Data (Features).
    """
    
    def __init__(self, forecast_horizon: int = None):
        if forecast_horizon is None:
            self.horizon = CONFIG.FORECAST_HORIZON
        else:
            self.horizon = forecast_horizon
        
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        # Simple ADX implementation
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        atr = tr.rolling(period).mean()
        # Handle zeros in atr
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr.replace(0, np.nan))
        
        sum_di = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / sum_di.replace(0, 1))
        adx = dx.rolling(period).mean()
        return adx.fillna(0)

    def label_regimes(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Label regimes based on future N-day returns and volatility.
        """
        df = prices.copy()
        
        # 1. Forward Looking Metrics (The Truth)
        # Shift(-N) means "The value N days in the future"
        
        # Future Volatility (N-day rolling std * sqrt(252))
        future_vol = df['close'].pct_change().rolling(self.horizon).std().shift(-self.horizon) * np.sqrt(252)
        
        # Future Drawdown (Min Low in next N days relative to Current Close)
        future_low = df['low'].rolling(self.horizon).min().shift(-self.horizon)
        future_drawdown = (future_low - df['close']) / df['close']
        
        # Future ADX (Trend Strength)
        current_adx = self._calculate_adx(df, 14)
        future_adx = current_adx.shift(-self.horizon)
        
        # Future Liquidity (Amihud)
        dollar_vol = df['close'] * df['volume']
        amihud = (df['close'].pct_change().abs() / dollar_vol).replace([np.inf, -np.inf], np.nan)
        # Rolling mean of Amihud 
        future_amihud = amihud.rolling(self.horizon).mean().shift(-self.horizon)
        
        # 2. Dynamic Thresholds (Rolling to handle non-stationarity)
        # BUT for Ground Truth, we can use Full Sample quantiles? 
        # No, strict walk-forward requires using only past data or adaptive dynamic thresholds.
        # But wait, we are labeling the *Training Data*. The label IS the ground truth.
        # We define "High Volatility" as the top X% of volatility *in that era*.
        # Let's use Expanding or Rolling Quantiles for the "Truth" to avoid lookahead bias in DEFINITION?
        # Actually, "Ground Truth" does not need to be tradeable. It describes what happened.
        # Ideally, we used a rolling median/quantile to define what was "High" at that time.
        
        # Use simple global quantile for this experiment as per config (or rolling if preferred).
        # Rolling is safer for long history.
        
        vol_high_thresh = future_vol.rolling(252).quantile(CONFIG.VOL_PERCENTILE)
        # Backfill
        vol_high_thresh = vol_high_thresh.fillna(method='bfill')
        
        liq_high_thresh = future_amihud.rolling(252).quantile(CONFIG.LIQ_PERCENTILE)
        liq_high_thresh = liq_high_thresh.fillna(method='bfill')
        
        # 3. Label: Volatility (Risk Off)
        # Risk Off if: Future Vol > Threshold OR Crash (Drawdown < -5% in 5 days)
        # Simplification: -5% in 5 days is ~ -1% per day.
        
        regime_vol = pd.Series(MarketRegime.NEUTRAL, index=df.index)
        mask_high_vol = (future_vol > vol_high_thresh) | (future_drawdown < -0.05)
        regime_vol[mask_high_vol] = MarketRegime.HIGH_VOL
        
        # 4. Label: Trend Quality
        # Robust if Future ADX > Config Threshold
        regime_trend = pd.Series(TrendQuality.NOISY, index=df.index)
        mask_adjust = (future_adx > CONFIG.TREND_ADX_THRESHOLD)
        regime_trend[mask_adjust] = TrendQuality.ROBUST
        
        # 5. Label: Liquidity (Safety)
        # Stressed if Future Amihud > Threshold
        regime_liq = pd.Series(LiquidityRegime.NORMAL, index=df.index)
        mask_liq = (future_amihud > liq_high_thresh)
        regime_liq[mask_liq] = LiquidityRegime.STRESSED
        
        # Combine
        df['regime_vol'] = regime_vol
        df['regime_trend'] = regime_trend
        df['regime_liq'] = regime_liq
        
        # Return df with labels
        return df[['regime_vol', 'regime_trend', 'regime_liq']]
