"""
Capital Allocator

Allocates capital across assets based on:
- Regime confidence per asset
- Asset volatility
- Correlation contribution

NO equal-weighting by default.
NO NEW ALPHA LOGIC.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Result of capital allocation."""
    symbol: str
    weight: float  # 0-1
    notional: float  # $ amount
    confidence: float
    volatility: float
    reason: str


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation."""
    total_capital: float
    allocations: Dict[str, AllocationResult]
    total_allocated: float
    cash_reserve: float


class CapitalAllocator:
    """
    Allocates capital based on regime confidence and risk metrics.
    
    Usage:
        allocator = CapitalAllocator()
        allocation = allocator.allocate(
            capital=100000,
            signals={"SPY": {"confidence": 0.7, "signal": "LONG"}},
            volatilities={"SPY": 0.15}
        )
    """
    
    def __init__(
        self,
        max_single_weight: float = 0.5,  # Max 50% in one asset
        min_confidence: float = 0.4,  # Don't allocate below this
        volatility_target: float = 0.15,  # Target portfolio vol
        cash_reserve_pct: float = 0.05  # Keep 5% cash
    ):
        self.max_single_weight = max_single_weight
        self.min_confidence = min_confidence
        self.volatility_target = volatility_target
        self.cash_reserve_pct = cash_reserve_pct
        
        logger.info("CapitalAllocator initialized")
    
    def calculate_weights(
        self,
        signals: Dict[str, Dict],  # symbol: {"confidence": float, "signal": str}
        volatilities: Dict[str, float]  # symbol: annualized vol
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights.
        
        Uses inverse volatility weighting scaled by confidence.
        """
        if not signals:
            return {}
        
        weights = {}
        
        for symbol, signal_info in signals.items():
            confidence = signal_info.get("confidence", 0.5)
            signal = signal_info.get("signal", "HOLD")
            
            # Skip HOLD or EXIT signals
            if signal not in ["ENTER_LONG", "LONG"]:
                weights[symbol] = 0.0
                continue
            
            # Skip low confidence
            if confidence < self.min_confidence:
                weights[symbol] = 0.0
                continue
            
            # Base weight from inverse volatility
            vol = volatilities.get(symbol, 0.20)  # Default 20%
            if vol > 0:
                inv_vol_weight = self.volatility_target / vol
            else:
                inv_vol_weight = 1.0
            
            # Scale by confidence
            weights[symbol] = inv_vol_weight * confidence
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Apply max single weight cap
        for symbol in weights:
            weights[symbol] = min(weights[symbol], self.max_single_weight)
        
        # Re-normalize after capping
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def allocate(
        self,
        capital: float,
        signals: Dict[str, Dict],
        volatilities: Dict[str, float]
    ) -> PortfolioAllocation:
        """
        Allocate capital across assets.
        
        Returns:
            PortfolioAllocation with per-asset allocations
        """
        # Reserve cash
        available = capital * (1 - self.cash_reserve_pct)
        
        # Calculate weights
        weights = self.calculate_weights(signals, volatilities)
        
        allocations = {}
        total_allocated = 0.0
        
        for symbol, weight in weights.items():
            notional = available * weight
            confidence = signals.get(symbol, {}).get("confidence", 0.5)
            vol = volatilities.get(symbol, 0.20)
            
            if weight > 0:
                reason = f"Confidence {confidence:.0%}, Vol {vol:.1%}"
            else:
                signal = signals.get(symbol, {}).get("signal", "HOLD")
                if signal not in ["ENTER_LONG", "LONG"]:
                    reason = f"Signal is {signal}"
                elif confidence < self.min_confidence:
                    reason = f"Confidence {confidence:.0%} < {self.min_confidence:.0%}"
                else:
                    reason = "Not allocated"
            
            allocations[symbol] = AllocationResult(
                symbol=symbol,
                weight=weight,
                notional=notional,
                confidence=confidence,
                volatility=vol,
                reason=reason
            )
            
            total_allocated += notional
        
        return PortfolioAllocation(
            total_capital=capital,
            allocations=allocations,
            total_allocated=total_allocated,
            cash_reserve=capital - total_allocated
        )
    
    def generate_allocation_report(
        self,
        allocation: PortfolioAllocation
    ) -> str:
        """Generate markdown allocation report."""
        content = f"""# Portfolio Allocation

**Total Capital**: ${allocation.total_capital:,.0f}
**Allocated**: ${allocation.total_allocated:,.0f}
**Cash Reserve**: ${allocation.cash_reserve:,.0f}

## Allocations

| Asset | Weight | Notional | Confidence | Vol | Reason |
|-------|--------|----------|------------|-----|--------|
"""
        for symbol, alloc in allocation.allocations.items():
            content += f"| {symbol} | {alloc.weight:.1%} | ${alloc.notional:,.0f} | {alloc.confidence:.0%} | {alloc.volatility:.1%} | {alloc.reason} |\n"
        
        return content


if __name__ == "__main__":
    allocator = CapitalAllocator()
    
    signals = {
        "SPY": {"confidence": 0.7, "signal": "ENTER_LONG"},
        "GLD": {"confidence": 0.6, "signal": "ENTER_LONG"},
        "TLT": {"confidence": 0.45, "signal": "HOLD"},
        "QQQ": {"confidence": 0.8, "signal": "ENTER_LONG"}
    }
    
    volatilities = {
        "SPY": 0.15,
        "GLD": 0.12,
        "TLT": 0.10,
        "QQQ": 0.20
    }
    
    allocation = allocator.allocate(100000, signals, volatilities)
    
    print(f"Total allocated: ${allocation.total_allocated:,.0f}")
    print(f"Cash reserve: ${allocation.cash_reserve:,.0f}")
    print()
    for symbol, alloc in allocation.allocations.items():
        print(f"{symbol}: {alloc.weight:.1%} (${alloc.notional:,.0f}) - {alloc.reason}")
