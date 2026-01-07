"""
Asset Selection Criteria

Defines eligibility requirements for portfolio inclusion:
- High liquidity (ADV > $1B)
- Clear volatility regimes
- Clean overnight execution
- Low short-borrow complexity

Initial candidates: SPY, GLD, TLT, QQQ

NO NEW ALPHA LOGIC. INDEPENDENT DECISIONS PER ASSET.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AssetEligibility:
    """Eligibility assessment for an asset."""
    symbol: str
    eligible: bool
    
    # Criteria scores
    liquidity_score: float  # 0-1
    regime_clarity_score: float  # 0-1
    execution_score: float  # 0-1
    complexity_score: float  # 0-1 (higher = simpler)
    
    # Details
    avg_daily_volume: float
    avg_spread_bps: float
    regime_consistency: float
    
    # Reasons
    rejection_reasons: List[str]


# Pre-approved universe
APPROVED_UNIVERSE = {
    "SPY": {"name": "S&P 500 ETF", "category": "equity", "leveraged": False},
    "GLD": {"name": "Gold ETF", "category": "commodity", "leveraged": False},
    "TLT": {"name": "20+ Year Treasury", "category": "fixed_income", "leveraged": False},
    "QQQ": {"name": "Nasdaq 100 ETF", "category": "equity", "leveraged": False},
    "IWM": {"name": "Russell 2000 ETF", "category": "equity", "leveraged": False},
    "EFA": {"name": "Developed Markets ETF", "category": "international", "leveraged": False},
}

# Explicitly excluded
EXCLUDED_ASSETS = {
    "TQQQ", "SQQQ", "UVXY", "VXX",  # Leveraged / Decay
    "SPXS", "SPXL",  # 3x leveraged
}


class AssetCriteria:
    """
    Evaluates assets for portfolio inclusion.
    
    Usage:
        criteria = AssetCriteria()
        eligibility = criteria.evaluate("SPY", price_data, volume_data)
    """
    
    def __init__(
        self,
        min_adv_usd: float = 1_000_000_000,  # $1B min daily volume
        max_spread_bps: float = 10.0,  # Max 10 bps spread
        min_regime_consistency: float = 0.6  # 60% regime consistency
    ):
        self.min_adv_usd = min_adv_usd
        self.max_spread_bps = max_spread_bps
        self.min_regime_consistency = min_regime_consistency
        
        logger.info("AssetCriteria initialized")
    
    def evaluate(
        self,
        symbol: str,
        prices: Optional[pd.DataFrame] = None,
        volume: Optional[pd.Series] = None,
        spreads: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None
    ) -> AssetEligibility:
        """
        Evaluate an asset for portfolio eligibility.
        
        Args:
            symbol: Asset symbol
            prices: OHLCV DataFrame
            volume: Daily volume series
            spreads: Bid-ask spread series (in bps)
            regime_labels: Regime label series for consistency check
        """
        rejection_reasons = []
        
        # Check if explicitly excluded
        if symbol in EXCLUDED_ASSETS:
            return AssetEligibility(
                symbol=symbol,
                eligible=False,
                liquidity_score=0,
                regime_clarity_score=0,
                execution_score=0,
                complexity_score=0,
                avg_daily_volume=0,
                avg_spread_bps=0,
                regime_consistency=0,
                rejection_reasons=["Explicitly excluded (leveraged/decay asset)"]
            )
        
        # Default values if no data provided
        adv = 0.0
        avg_spread = 0.0
        regime_consistency = 0.0
        
        # 1. Liquidity check
        if volume is not None and prices is not None:
            # Calculate ADV in USD
            if 'Close' in prices.columns:
                adv = (volume * prices['Close']).mean()
            else:
                adv = volume.mean() * 100  # Estimate
        
        if adv < self.min_adv_usd and adv > 0:
            rejection_reasons.append(f"Insufficient liquidity: ${adv/1e9:.2f}B < ${self.min_adv_usd/1e9:.2f}B")
        
        liquidity_score = min(1.0, adv / (self.min_adv_usd * 2)) if adv > 0 else 0.5
        
        # 2. Spread check
        if spreads is not None:
            avg_spread = spreads.mean()
            if avg_spread > self.max_spread_bps:
                rejection_reasons.append(f"High spread: {avg_spread:.1f} bps > {self.max_spread_bps} bps")
        
        execution_score = max(0, 1 - avg_spread / (self.max_spread_bps * 2)) if avg_spread > 0 else 0.7
        
        # 3. Regime clarity check
        if regime_labels is not None:
            # Measure regime consistency (how often it stays in same regime)
            if len(regime_labels) > 1:
                same_regime = (regime_labels == regime_labels.shift(1)).mean()
                regime_consistency = float(same_regime)
            if regime_consistency < self.min_regime_consistency:
                rejection_reasons.append(f"Low regime clarity: {regime_consistency:.1%}")
        
        regime_clarity_score = regime_consistency if regime_consistency > 0 else 0.7
        
        # 4. Complexity check (simple = good)
        if symbol in APPROVED_UNIVERSE:
            asset_info = APPROVED_UNIVERSE[symbol]
            complexity_score = 1.0 if not asset_info.get("leveraged", False) else 0.3
        else:
            complexity_score = 0.5  # Unknown
            rejection_reasons.append("Not in pre-approved universe")
        
        eligible = len(rejection_reasons) == 0
        
        return AssetEligibility(
            symbol=symbol,
            eligible=eligible,
            liquidity_score=liquidity_score,
            regime_clarity_score=regime_clarity_score,
            execution_score=execution_score,
            complexity_score=complexity_score,
            avg_daily_volume=adv,
            avg_spread_bps=avg_spread,
            regime_consistency=regime_consistency,
            rejection_reasons=rejection_reasons
        )
    
    def evaluate_universe(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, AssetEligibility]:
        """Evaluate entire universe (quick check without data)."""
        if symbols is None:
            symbols = list(APPROVED_UNIVERSE.keys())
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.evaluate(symbol)
        
        return results
    
    def get_eligible_assets(self) -> List[str]:
        """Get list of eligible assets from approved universe."""
        return list(APPROVED_UNIVERSE.keys())


if __name__ == "__main__":
    criteria = AssetCriteria()
    
    # Check approved universe
    results = criteria.evaluate_universe()
    
    print("Asset Eligibility:")
    for symbol, result in results.items():
        status = "✅" if result.eligible else "❌"
        print(f"{status} {symbol}: {result.rejection_reasons if not result.eligible else 'Approved'}")
