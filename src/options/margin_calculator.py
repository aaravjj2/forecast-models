"""
Margin Calculator

Pre-trade and post-trade margin analysis:
- Calculate margin impact before order
- Reject orders that breach bounds
- Daily maintenance margin checks

Paper-first. Model real margin requirements.
"""

import logging
from datetime import datetime, timezone, date
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionType(Enum):
    LONG_STOCK = "long_stock"
    SHORT_STOCK = "short_stock"
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    IRON_CONDOR = "iron_condor"
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"


@dataclass
class MarginRequirement:
    """Margin requirement for a position."""
    position_type: PositionType
    initial_margin: float
    maintenance_margin: float
    buying_power_impact: float
    margin_type: str  # "Reg-T", "Portfolio Margin", "Cash"


@dataclass
class AccountMarginStatus:
    """Current account margin status."""
    equity: float
    margin_used: float
    margin_available: float
    buying_power: float
    maintenance_excess: float
    margin_ratio: float  # margin_used / equity


class MarginCalculator:
    """
    Calculates margin requirements for options positions.
    
    Usage:
        calc = MarginCalculator(account_equity=100000)
        req = calc.calculate_requirement("covered_call", ...)
        allowed, reason = calc.can_open_position(req)
    """
    
    # Reg-T margin rates
    REG_T_RATES = {
        "stock": 0.50,  # 50% initial margin for stocks
        "short_naked_call": 0.20,  # 20% of underlying + premium
        "short_naked_put": 0.20,  # 20% of underlying + premium
        "spread": 1.0,  # Width of spread
        "covered_call": 0.0,  # No additional margin
        "protective_put": 0.0,  # No additional margin (long option)
    }
    
    def __init__(
        self,
        account_equity: float = 100000,
        max_margin_ratio: float = 0.60,  # Max 60% margin utilization
        min_maintenance_excess: float = 5000  # Min $5k buffer
    ):
        self.account_equity = account_equity
        self.max_margin_ratio = max_margin_ratio
        self.min_maintenance_excess = min_maintenance_excess
        
        self.margin_used = 0.0
        self.positions: List[Dict] = []
        
        logger.info(f"MarginCalculator initialized: equity=${account_equity:,.0f}")
    
    def calculate_requirement(
        self,
        position_type: PositionType,
        underlying_price: float,
        strike: float,
        premium: float,
        contracts: int = 1,
        spread_width: Optional[float] = None
    ) -> MarginRequirement:
        """
        Calculate margin requirement for a position.
        """
        notional = underlying_price * 100 * contracts
        premium_total = premium * 100 * contracts
        
        if position_type == PositionType.LONG_CALL or position_type == PositionType.LONG_PUT:
            # Long options: only cost
            initial_margin = premium_total
            maintenance_margin = 0
            buying_power = premium_total
            margin_type = "Cash"
        
        elif position_type == PositionType.COVERED_CALL:
            # Covered call: stock margin + no additional for call
            initial_margin = notional * self.REG_T_RATES["stock"]
            maintenance_margin = notional * 0.25  # 25% maintenance
            buying_power = initial_margin - premium_total  # Premium offsets
            margin_type = "Reg-T"
        
        elif position_type == PositionType.PROTECTIVE_PUT:
            # Protective put: stock margin + put cost
            initial_margin = notional * self.REG_T_RATES["stock"] + premium_total
            maintenance_margin = notional * 0.25
            buying_power = initial_margin
            margin_type = "Reg-T"
        
        elif position_type == PositionType.SHORT_CALL:
            # Naked short call: 20% of underlying + premium + OTM amount
            otm_amount = max(0, underlying_price - strike) * 100 * contracts
            initial_margin = (notional * 0.20 + premium_total + otm_amount)
            maintenance_margin = initial_margin * 0.8
            buying_power = initial_margin
            margin_type = "Reg-T"
        
        elif position_type == PositionType.SHORT_PUT:
            # Naked short put: 20% of underlying + premium - OTM amount
            otm_amount = max(0, strike - underlying_price) * 100 * contracts
            initial_margin = max(
                notional * 0.10,  # Min 10%
                notional * 0.20 + premium_total - otm_amount
            )
            maintenance_margin = initial_margin * 0.8
            buying_power = initial_margin
            margin_type = "Reg-T"
        
        elif position_type in [PositionType.IRON_CONDOR, PositionType.PUT_SPREAD, PositionType.CALL_SPREAD]:
            # Defined risk spreads: max loss = spread width
            spread_width = spread_width or 5.0
            max_loss = spread_width * 100 * contracts
            initial_margin = max_loss - premium_total  # Net debit/credit
            maintenance_margin = max_loss
            buying_power = max_loss
            margin_type = "Reg-T"
        
        else:
            # Default: full notional
            initial_margin = notional
            maintenance_margin = notional * 0.25
            buying_power = notional
            margin_type = "Cash"
        
        return MarginRequirement(
            position_type=position_type,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            buying_power_impact=buying_power,
            margin_type=margin_type
        )
    
    def can_open_position(self, requirement: MarginRequirement) -> tuple:
        """
        Check if position can be opened within margin limits.
        
        Returns (allowed, reason).
        """
        new_margin = self.margin_used + requirement.initial_margin
        new_ratio = new_margin / self.account_equity
        
        if new_ratio > self.max_margin_ratio:
            return False, f"Margin ratio {new_ratio:.1%} exceeds {self.max_margin_ratio:.1%} limit"
        
        new_excess = self.account_equity - new_margin
        if new_excess < self.min_maintenance_excess:
            return False, f"Maintenance excess ${new_excess:,.0f} below ${self.min_maintenance_excess:,.0f} minimum"
        
        return True, "Position allowed"
    
    def add_position(self, requirement: MarginRequirement):
        """Add position to account."""
        self.margin_used += requirement.initial_margin
        self.positions.append({
            "type": requirement.position_type.value,
            "initial": requirement.initial_margin,
            "maintenance": requirement.maintenance_margin
        })
        logger.info(f"Added position: margin now ${self.margin_used:,.0f}")
    
    def get_account_status(self) -> AccountMarginStatus:
        """Get current account margin status."""
        return AccountMarginStatus(
            equity=self.account_equity,
            margin_used=self.margin_used,
            margin_available=self.account_equity * self.max_margin_ratio - self.margin_used,
            buying_power=self.account_equity - self.margin_used,
            maintenance_excess=self.account_equity - self.margin_used - self.min_maintenance_excess,
            margin_ratio=self.margin_used / self.account_equity if self.account_equity > 0 else 0
        )
    
    def check_maintenance_margin(self) -> tuple:
        """
        Daily maintenance margin check.
        
        Returns (ok, margin_call_amount).
        """
        total_maintenance = sum(p["maintenance"] for p in self.positions)
        excess = self.account_equity - total_maintenance
        
        if excess < 0:
            return False, abs(excess)
        
        return True, 0
    
    def simulate_move(self, underlying_change_pct: float) -> AccountMarginStatus:
        """
        Simulate margin impact from underlying move.
        """
        # Simplified: assume margin requirements scale with underlying
        new_margin = self.margin_used * (1 + underlying_change_pct)
        new_ratio = new_margin / self.account_equity
        
        return AccountMarginStatus(
            equity=self.account_equity,
            margin_used=new_margin,
            margin_available=self.account_equity * self.max_margin_ratio - new_margin,
            buying_power=self.account_equity - new_margin,
            maintenance_excess=self.account_equity - new_margin - self.min_maintenance_excess,
            margin_ratio=new_ratio
        )


if __name__ == "__main__":
    calc = MarginCalculator(account_equity=100000)
    
    # Covered call
    req = calc.calculate_requirement(
        PositionType.COVERED_CALL,
        underlying_price=450,
        strike=460,
        premium=3.50,
        contracts=1
    )
    print(f"Covered Call margin: ${req.initial_margin:,.0f}")
    
    allowed, reason = calc.can_open_position(req)
    print(f"Allowed: {allowed} - {reason}")
    
    if allowed:
        calc.add_position(req)
    
    # Iron condor
    req2 = calc.calculate_requirement(
        PositionType.IRON_CONDOR,
        underlying_price=450,
        strike=440,
        premium=2.00,
        contracts=1,
        spread_width=5
    )
    print(f"\nIron Condor margin: ${req2.initial_margin:,.0f}")
    
    # Status
    status = calc.get_account_status()
    print(f"\nAccount Status:")
    print(f"  Equity: ${status.equity:,.0f}")
    print(f"  Margin Used: ${status.margin_used:,.0f}")
    print(f"  Margin Ratio: {status.margin_ratio:.1%}")
