"""
Covered Calls Strategy

During Risk-On + Trend regimes:
- Sell short-dated covered calls for income
- Cap upside but generate premium

Paper-first. Model assignment risk.
"""

import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoveredCallPosition:
    """A covered call position."""
    underlying: str
    shares: int
    underlying_cost: float
    call_strike: float
    call_expiration: date
    call_premium: float  # Premium received
    max_profit: float  # Max profit if assigned
    breakeven: float


@dataclass
class CoveredCallResult:
    """Result of covered call simulation."""
    underlying_return: float
    covered_return: float
    assigned: bool
    premium_kept: float
    upside_capped: float  # Profit given up due to cap


class CoveredCallStrategy:
    """
    Implements covered call overlay for Risk-On regimes.
    
    Usage:
        strategy = CoveredCallStrategy()
        position = strategy.sell_call("SPY", 100, delta=0.30)
        result = strategy.simulate(position, underlying_price_change=0.05)
    """
    
    def __init__(
        self,
        data_fetcher=None,
        target_delta: float = 0.30,  # 30-delta calls (OTM)
        max_dte: int = 30,  # Max 30 DTE for time decay
        min_premium_pct: float = 0.005  # Min 0.5% premium
    ):
        from .data_fetcher import OptionsDataFetcher
        self.data_fetcher = data_fetcher or OptionsDataFetcher()
        self.target_delta = target_delta
        self.max_dte = max_dte
        self.min_premium_pct = min_premium_pct
        
        logger.info(f"CoveredCallStrategy initialized (delta={target_delta})")
    
    def find_optimal_call(
        self,
        symbol: str,
        underlying_price: float,
        target_delta: Optional[float] = None
    ) -> tuple:
        """
        Find optimal call to sell.
        
        Returns (contract, expiration, premium).
        """
        target_delta = target_delta or self.target_delta
        
        expirations = self.data_fetcher.get_expirations(symbol)
        today = date.today()
        
        # Find expiration within max_dte
        valid_exps = [e for e in expirations if 7 <= (e - today).days <= self.max_dte]
        
        if not valid_exps:
            valid_exps = [e for e in expirations if (e - today).days <= self.max_dte]
        
        if not valid_exps:
            logger.warning("No valid expirations found")
            return None, None, 0
        
        # Use first valid expiration
        expiration = valid_exps[0]
        chain = self.data_fetcher.get_chain(symbol, expiration.strftime("%Y-%m-%d"))
        
        # Find OTM call (above current price)
        target_strike = underlying_price * (1 + target_delta * 0.10)  # ~3% OTM for 30-delta
        
        otm_calls = [c for c in chain.calls if c.strike > underlying_price]
        if not otm_calls:
            return None, None, 0
        
        call = min(otm_calls, key=lambda c: abs(c.strike - target_strike))
        
        # Premium (use bid for selling)
        premium = call.bid if call.bid > 0 else call.last * 0.95
        
        return call, expiration, premium
    
    def sell_call(
        self,
        symbol: str,
        shares: int,
        underlying_price: Optional[float] = None,
        target_delta: Optional[float] = None
    ) -> CoveredCallPosition:
        """
        Sell covered call against shares.
        
        Args:
            symbol: Underlying symbol
            shares: Number of shares held
            underlying_price: Current price
            target_delta: Delta target for calls
        """
        if underlying_price is None:
            exps = self.data_fetcher.get_expirations(symbol)
            if exps:
                chain = self.data_fetcher.get_chain(symbol, exps[0].strftime("%Y-%m-%d"))
                underlying_price = chain.underlying_price
            else:
                underlying_price = 100.0
        
        call, expiration, premium_per_share = self.find_optimal_call(
            symbol, underlying_price, target_delta
        )
        
        if call is None:
            logger.error("Could not find suitable call")
            return None
        
        # Calculate position
        contracts = shares // 100
        total_premium = premium_per_share * contracts * 100
        
        # Max profit = premium + (strike - current price)
        max_profit = total_premium + (call.strike - underlying_price) * shares
        
        # Breakeven = current price - premium
        breakeven = underlying_price - premium_per_share
        
        position = CoveredCallPosition(
            underlying=symbol,
            shares=shares,
            underlying_cost=underlying_price,
            call_strike=call.strike,
            call_expiration=expiration,
            call_premium=total_premium,
            max_profit=max_profit,
            breakeven=breakeven
        )
        
        logger.info(f"Covered call: {contracts}x {call.strike}C @ ${premium_per_share:.2f}")
        logger.info(f"Premium received: ${total_premium:.2f} ({total_premium / (underlying_price * shares) * 100:.2f}%)")
        
        return position
    
    def simulate(
        self,
        position: CoveredCallPosition,
        underlying_returns: List[float]
    ) -> List[CoveredCallResult]:
        """
        Simulate covered call payoff across scenarios.
        """
        results = []
        
        initial_price = position.underlying_cost
        position_value = initial_price * position.shares
        premium_pct = position.call_premium / position_value
        
        for ret in underlying_returns:
            final_price = initial_price * (1 + ret)
            
            # Underlying P&L
            underlying_pnl = (final_price - initial_price) * position.shares
            
            # Call payoff (simplified - at expiration)
            assigned = final_price >= position.call_strike
            
            if assigned:
                # Shares called away at strike
                effective_pnl = (position.call_strike - initial_price) * position.shares + position.call_premium
                upside_capped = (final_price - position.call_strike) * position.shares
            else:
                # Keep shares + premium
                effective_pnl = underlying_pnl + position.call_premium
                upside_capped = 0
            
            covered_return = effective_pnl / position_value
            
            results.append(CoveredCallResult(
                underlying_return=ret,
                covered_return=covered_return,
                assigned=assigned,
                premium_kept=position.call_premium if not assigned else position.call_premium,
                upside_capped=upside_capped
            ))
        
        return results
    
    def backtest(
        self,
        symbol: str,
        returns: List[float],
        regime_signals: List[str],
        shares: int = 100
    ) -> Dict:
        """
        Backtest covered calls during Risk-On periods.
        """
        total_pnl = 0.0
        premiums_collected = 0.0
        times_assigned = 0
        upside_given_up = 0.0
        
        position = None
        days_in_position = 0
        
        for i, (ret, regime) in enumerate(zip(returns, regime_signals)):
            if regime == "RISK_ON" and position is None:
                # Sell covered call
                position = self.sell_call(symbol, shares)
                if position:
                    premiums_collected += position.call_premium
                days_in_position = 0
            
            elif regime != "RISK_ON" and position is not None:
                # Close position (let expire or buy back)
                position = None
                days_in_position = 0
            
            if position is not None:
                days_in_position += 1
                
                # Check for assignment (simplified)
                if ret > 0.05:  # Big up day, likely assigned
                    times_assigned += 1
                    capped = ret - 0.03  # Lost upside above strike
                    upside_given_up += capped * shares * 100
            
            total_pnl += ret * shares * 100
        
        return {
            "total_return": total_pnl,
            "premiums_collected": premiums_collected,
            "net_return": total_pnl + premiums_collected - upside_given_up,
            "times_assigned": times_assigned,
            "upside_given_up": upside_given_up,
            "avg_premium": premiums_collected / max(1, times_assigned + 1)
        }


if __name__ == "__main__":
    strategy = CoveredCallStrategy()
    
    # Sell covered call
    position = strategy.sell_call("SPY", 100)
    
    if position:
        print(f"\nPosition: {position.shares} shares + short {position.call_strike}C")
        print(f"Premium: ${position.call_premium:.2f}")
        print(f"Max profit: ${position.max_profit:.2f}")
        print(f"Breakeven: ${position.breakeven:.2f}")
        
        # Simulate scenarios
        scenarios = [-0.10, -0.05, 0.0, 0.03, 0.05, 0.10]
        results = strategy.simulate(position, scenarios)
        
        print("\nScenario Analysis:")
        for ret, result in zip(scenarios, results):
            assigned = "ASSIGNED" if result.assigned else ""
            print(f"  Market {ret*100:+.0f}% → Covered {result.covered_return*100:+.1f}% {assigned}")
