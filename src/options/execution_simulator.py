"""
Options Execution Simulator

Realistic execution modeling:
- Wide bid-ask spreads
- Multi-leg fill failures
- Partial fills
- Assignment risk
- Early assignment

Paper-first. Model real-world execution challenges.
"""

import logging
from datetime import datetime, timezone, date
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FillStatus(Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class OptionFill:
    """Simulated option fill."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    filled_quantity: int
    limit_price: float
    fill_price: float
    slippage_cents: float
    status: FillStatus
    timestamp: str


@dataclass
class MultiLegFill:
    """Multi-leg order fill result."""
    legs: List[OptionFill]
    all_filled: bool
    total_slippage: float
    total_cost: float


@dataclass
class AssignmentEvent:
    """Option assignment event."""
    symbol: str
    strike: float
    option_type: str
    assigned_shares: int
    exercise_price: float
    cash_impact: float
    timestamp: str


class OptionsExecutionSimulator:
    """
    Simulates realistic options execution.
    
    Usage:
        sim = OptionsExecutionSimulator()
        fill = sim.simulate_fill(symbol, side, qty, price)
        assignment = sim.check_assignment(position, underlying_price)
    """
    
    def __init__(
        self,
        base_spread_bps: float = 50.0,  # 50 bps base spread
        fill_rate: float = 0.85,  # 85% fill probability
        partial_fill_rate: float = 0.10,  # 10% partial fills
        seed: Optional[int] = None
    ):
        self.base_spread_bps = base_spread_bps
        self.fill_rate = fill_rate
        self.partial_fill_rate = partial_fill_rate
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"OptionsExecutionSimulator initialized (spread={base_spread_bps}bps)")
    
    def calculate_spread(
        self,
        option_price: float,
        volume: int,
        dte: int
    ) -> float:
        """
        Calculate realistic bid-ask spread.
        
        Width depends on:
        - Option price (penny-wide for high volume, wider for low)
        - Volume (higher volume = tighter)
        - DTE (near-term = tighter)
        """
        # Base spread as % of price
        base_pct = self.base_spread_bps / 10000
        
        # Volume adjustment (lower volume = wider)
        volume_factor = max(0.5, min(2.0, 10000 / max(1, volume)))
        
        # DTE adjustment (more DTE = wider)
        dte_factor = max(0.8, min(1.5, dte / 30))
        
        spread_pct = base_pct * volume_factor * dte_factor
        spread = option_price * spread_pct
        
        # Minimum spread of $0.01 (penny)
        return max(0.01, spread)
    
    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float,
        market_bid: float,
        market_ask: float,
        volume: int = 1000
    ) -> OptionFill:
        """
        Simulate single-leg option fill.
        """
        spread = market_ask - market_bid
        mid = (market_bid + market_ask) / 2
        
        # Determine if order fills
        roll = np.random.random()
        
        if side == "buy":
            # Buying: need to hit ask or better
            if limit_price < market_ask and roll > self.fill_rate:
                return OptionFill(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    limit_price=limit_price,
                    fill_price=0,
                    slippage_cents=0,
                    status=FillStatus.REJECTED,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            # Fill price between mid and ask
            fill_price = mid + np.random.uniform(0, spread * 0.5)
            fill_price = min(fill_price, limit_price)
            slippage = (fill_price - mid) * 100
            
        else:  # sell
            # Selling: need to hit bid or better
            if limit_price > market_bid and roll > self.fill_rate:
                return OptionFill(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    limit_price=limit_price,
                    fill_price=0,
                    slippage_cents=0,
                    status=FillStatus.REJECTED,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            # Fill price between bid and mid
            fill_price = mid - np.random.uniform(0, spread * 0.5)
            fill_price = max(fill_price, limit_price)
            slippage = (mid - fill_price) * 100
        
        # Partial fill check
        if np.random.random() < self.partial_fill_rate:
            filled_qty = int(quantity * np.random.uniform(0.3, 0.9))
            status = FillStatus.PARTIAL
        else:
            filled_qty = quantity
            status = FillStatus.FILLED
        
        return OptionFill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            filled_quantity=filled_qty,
            limit_price=limit_price,
            fill_price=round(fill_price, 2),
            slippage_cents=round(slippage, 2),
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def simulate_multi_leg_fill(
        self,
        legs: List[Dict]
    ) -> MultiLegFill:
        """
        Simulate multi-leg order (spread, condor, etc.).
        
        Multi-leg orders have legging risk - some legs may fill while others don't.
        """
        fills = []
        total_slippage = 0
        total_cost = 0
        
        for leg in legs:
            fill = self.simulate_fill(
                symbol=leg["symbol"],
                side=leg["side"],
                quantity=leg["quantity"],
                limit_price=leg["limit_price"],
                market_bid=leg.get("bid", leg["limit_price"] * 0.95),
                market_ask=leg.get("ask", leg["limit_price"] * 1.05),
                volume=leg.get("volume", 1000)
            )
            fills.append(fill)
            
            if fill.status == FillStatus.FILLED or fill.status == FillStatus.PARTIAL:
                total_slippage += fill.slippage_cents * fill.filled_quantity
                
                if leg["side"] == "buy":
                    total_cost += fill.fill_price * fill.filled_quantity * 100
                else:
                    total_cost -= fill.fill_price * fill.filled_quantity * 100
        
        all_filled = all(f.status == FillStatus.FILLED for f in fills)
        
        return MultiLegFill(
            legs=fills,
            all_filled=all_filled,
            total_slippage=total_slippage,
            total_cost=total_cost
        )
    
    def check_assignment_risk(
        self,
        option_type: str,
        strike: float,
        underlying_price: float,
        dte: int,
        is_short: bool
    ) -> Tuple[float, str]:
        """
        Check assignment risk for short options.
        
        Returns (probability, reason).
        """
        if not is_short:
            return 0.0, "Long options cannot be assigned"
        
        if option_type == "call":
            itm = underlying_price - strike
            if itm <= 0:
                return 0.0, "Call is OTM"
            
            # Deep ITM calls near expiration have high assignment risk
            itm_pct = itm / underlying_price
            time_factor = max(0.1, 1 - dte / 30)
            prob = min(0.95, itm_pct * 10 * time_factor)
            
            if prob > 0.5:
                return prob, f"Deep ITM ({itm_pct:.1%}), high assignment risk"
            return prob, f"ITM ({itm_pct:.1%}), moderate assignment risk"
        
        else:  # put
            itm = strike - underlying_price
            if itm <= 0:
                return 0.0, "Put is OTM"
            
            itm_pct = itm / underlying_price
            time_factor = max(0.1, 1 - dte / 30)
            prob = min(0.90, itm_pct * 8 * time_factor)
            
            if prob > 0.5:
                return prob, f"Deep ITM ({itm_pct:.1%}), high assignment risk"
            return prob, f"ITM ({itm_pct:.1%}), moderate assignment risk"
    
    def simulate_assignment(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        contracts: int,
        underlying_price: float
    ) -> AssignmentEvent:
        """
        Simulate option assignment.
        """
        shares = contracts * 100
        
        if option_type == "call":
            # Short call: must sell shares at strike
            cash_impact = strike * shares  # Receive cash
        else:
            # Short put: must buy shares at strike
            cash_impact = -strike * shares  # Pay cash
        
        return AssignmentEvent(
            symbol=symbol,
            strike=strike,
            option_type=option_type,
            assigned_shares=shares,
            exercise_price=strike,
            cash_impact=cash_impact,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


if __name__ == "__main__":
    sim = OptionsExecutionSimulator(seed=42)
    
    # Single leg fill
    fill = sim.simulate_fill(
        symbol="SPY_2026-01-17_450C",
        side="buy",
        quantity=1,
        limit_price=5.50,
        market_bid=5.40,
        market_ask=5.60
    )
    print(f"Single leg fill: {fill.status.value}")
    print(f"  Fill price: ${fill.fill_price:.2f}")
    print(f"  Slippage: {fill.slippage_cents:.0f} cents")
    
    # Multi-leg (iron condor)
    legs = [
        {"symbol": "SPY_430P", "side": "buy", "quantity": 1, "limit_price": 1.50, "bid": 1.45, "ask": 1.55},
        {"symbol": "SPY_435P", "side": "sell", "quantity": 1, "limit_price": 2.00, "bid": 1.95, "ask": 2.05},
        {"symbol": "SPY_465C", "side": "sell", "quantity": 1, "limit_price": 2.00, "bid": 1.95, "ask": 2.05},
        {"symbol": "SPY_470C", "side": "buy", "quantity": 1, "limit_price": 1.50, "bid": 1.45, "ask": 1.55},
    ]
    
    multi = sim.simulate_multi_leg_fill(legs)
    print(f"\nMulti-leg fill: {'All filled' if multi.all_filled else 'Partial/Failed'}")
    print(f"  Total slippage: {multi.total_slippage:.0f} cents")
    print(f"  Net cost: ${multi.total_cost:.0f}")
    
    # Assignment risk
    prob, reason = sim.check_assignment_risk("put", 440, 435, 5, is_short=True)
    print(f"\nAssignment risk: {prob:.1%} - {reason}")
