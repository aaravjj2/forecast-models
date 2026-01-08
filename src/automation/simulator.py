"""
Simulator

Full simulation mode with:
- Real-time market data
- Simulated fills using slippage model
- Toggle: simulation → paper → live

Paper-first. No live orders unless explicitly enabled.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Literal
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulatedFill:
    """Simulated order fill."""
    symbol: str
    side: str
    quantity: float
    expected_price: float
    filled_price: float
    slippage_bps: float
    commission: float
    timestamp: str


class SlippageModel:
    """
    Slippage simulation model.
    
    Models slippage as function of:
    - Order size relative to ADV
    - Market volatility
    - Random execution noise
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        size_impact_factor: float = 0.1,  # bps per % of ADV
        volatility_factor: float = 0.5,
        seed: Optional[int] = None
    ):
        self.base_slippage = base_slippage_bps
        self.size_impact = size_impact_factor
        self.vol_factor = volatility_factor
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"SlippageModel initialized: base={base_slippage_bps}bps")
    
    def calculate_slippage(
        self,
        order_size: float,
        adv: float,
        volatility: float,
        side: str
    ) -> float:
        """
        Calculate expected slippage in bps.
        
        Args:
            order_size: Number of shares
            adv: Average daily volume
            volatility: Annualized volatility
            side: "buy" or "sell"
        
        Returns:
            Slippage in basis points (positive = unfavorable)
        """
        # Size impact
        pct_of_adv = order_size / adv if adv > 0 else 0.01
        size_slippage = self.size_impact * (pct_of_adv * 100)
        
        # Volatility impact
        vol_slippage = self.vol_factor * (volatility * 100)
        
        # Random noise (execution uncertainty)
        noise = np.random.normal(0, 2)  # 2 bps stdev
        
        # Total slippage
        total_slippage = self.base_slippage + size_slippage + vol_slippage + noise
        
        # Ensure non-negative
        return max(0, total_slippage)


class TradeSimulator:
    """
    Simulates trade execution with realistic fills.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0
    ):
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        
        # Historical data cache
        self.price_cache: Dict[str, float] = {}
        self.adv_cache: Dict[str, float] = {}
        self.vol_cache: Dict[str, float] = {}
        
        logger.info("TradeSimulator initialized")
    
    def set_market_data(
        self,
        symbol: str,
        price: float,
        adv: float,
        volatility: float
    ):
        """Set current market data for symbol."""
        self.price_cache[symbol] = price
        self.adv_cache[symbol] = adv
        self.vol_cache[symbol] = volatility
    
    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: Optional[float] = None
    ) -> SimulatedFill:
        """
        Simulate order fill with slippage.
        """
        expected_price = self.price_cache.get(symbol, 100.0)
        adv = self.adv_cache.get(symbol, 1_000_000)
        volatility = self.vol_cache.get(symbol, 0.20)
        
        # Calculate slippage
        slippage_bps = self.slippage_model.calculate_slippage(
            quantity, adv, volatility, side
        )
        
        # Apply slippage to price
        slippage_pct = slippage_bps / 10000
        
        if side == "buy":
            filled_price = expected_price * (1 + slippage_pct)
        else:
            filled_price = expected_price * (1 - slippage_pct)
        
        # Check limit price
        if limit_price is not None:
            if side == "buy" and filled_price > limit_price:
                # Would not fill at this price
                logger.warning(f"Limit order would not fill: {filled_price:.2f} > {limit_price:.2f}")
            elif side == "sell" and filled_price < limit_price:
                logger.warning(f"Limit order would not fill: {filled_price:.2f} < {limit_price:.2f}")
        
        # Calculate commission
        commission = max(self.min_commission, quantity * self.commission_per_share)
        
        return SimulatedFill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=expected_price,
            filled_price=filled_price,
            slippage_bps=slippage_bps,
            commission=commission,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class ExecutionModeManager:
    """
    Manages execution mode transitions:
    simulation → paper → live
    """
    
    MODES = ["simulation", "paper", "live"]
    
    def __init__(self, initial_mode: str = "simulation"):
        self._mode = initial_mode
        self._mode_history = [(initial_mode, datetime.now(timezone.utc).isoformat())]
        
        logger.info(f"ExecutionModeManager: starting in {initial_mode} mode")
    
    @property
    def mode(self) -> str:
        return self._mode
    
    def can_transition_to(self, target_mode: str) -> tuple:
        """
        Check if transition is allowed.
        Returns (allowed, reason).
        """
        if target_mode not in self.MODES:
            return False, f"Invalid mode: {target_mode}"
        
        current_idx = self.MODES.index(self._mode)
        target_idx = self.MODES.index(target_mode)
        
        # Can always go backwards
        if target_idx < current_idx:
            return True, "Downgrade allowed"
        
        # Forward transitions require checks
        if target_mode == "live":
            if not os.environ.get("ENABLE_LIVE_TRADING"):
                return False, "ENABLE_LIVE_TRADING not set"
            if not os.environ.get("CONFIRM_LIVE_TRADING"):
                return False, "CONFIRM_LIVE_TRADING not set"
        
        return True, "Transition allowed"
    
    def transition_to(self, target_mode: str) -> tuple:
        """
        Transition to target mode.
        Returns (success, message).
        """
        allowed, reason = self.can_transition_to(target_mode)
        
        if not allowed:
            logger.warning(f"Mode transition blocked: {reason}")
            return False, reason
        
        old_mode = self._mode
        self._mode = target_mode
        self._mode_history.append((target_mode, datetime.now(timezone.utc).isoformat()))
        
        logger.info(f"Mode transition: {old_mode} → {target_mode}")
        return True, f"Transitioned from {old_mode} to {target_mode}"
    
    def get_history(self) -> list:
        """Get mode transition history."""
        return self._mode_history


if __name__ == "__main__":
    # Quick test
    simulator = TradeSimulator()
    
    # Set market data
    simulator.set_market_data("SPY", 450.0, 50_000_000, 0.15)
    
    # Simulate fills
    for _ in range(5):
        fill = simulator.simulate_fill("SPY", "buy", 100)
        print(f"Fill: {fill.filled_price:.2f} (slippage: {fill.slippage_bps:.1f}bps)")
    
    # Mode manager
    mode_manager = ExecutionModeManager()
    print(f"\nCurrent mode: {mode_manager.mode}")
    
    success, msg = mode_manager.transition_to("live")
    print(f"Live transition: {success} - {msg}")
