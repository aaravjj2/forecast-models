"""
Protective Puts Strategy

During Risk-Off regimes:
- Buy X-delta put for tail protection
- Simulate cost and payoff

Paper-first. Verify protection vs cost.
"""

import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProtectivePutPosition:
    """A protective put position."""
    underlying: str
    shares: int
    put_strike: float
    put_expiration: date
    put_cost: float  # Total cost for puts
    delta: float
    protection_level: float  # % of downside protected


@dataclass
class ProtectivePutResult:
    """Result of protective put simulation."""
    underlying_return: float
    protected_return: float
    max_loss: float
    put_pnl: float
    total_cost_pct: float


class ProtectivePutStrategy:
    """
    Implements protective put overlay for Risk-Off regimes.
    
    Usage:
        strategy = ProtectivePutStrategy()
        position = strategy.buy_protection("SPY", 100, delta=0.30)
        result = strategy.simulate(position, underlying_price_change=-0.10)
    """
    
    def __init__(
        self,
        data_fetcher=None,
        max_cost_pct: float = 0.02,  # Max 2% of position for protection
        target_delta: float = 0.30,  # 30-delta puts
        min_dte: int = 21  # Minimum 21 DTE
    ):
        from .data_fetcher import OptionsDataFetcher
        self.data_fetcher = data_fetcher or OptionsDataFetcher()
        self.max_cost_pct = max_cost_pct
        self.target_delta = target_delta
        self.min_dte = min_dte
        
        logger.info(f"ProtectivePutStrategy initialized (delta={target_delta})")
    
    def find_optimal_put(
        self,
        symbol: str,
        underlying_price: float,
        target_delta: Optional[float] = None
    ) -> tuple:
        """
        Find optimal put for protection.
        
        Returns (contract, expiration, cost).
        """
        target_delta = target_delta or self.target_delta
        
        expirations = self.data_fetcher.get_expirations(symbol)
        
        # Filter expirations with minimum DTE
        today = date.today()
        valid_exps = [e for e in expirations if (e - today).days >= self.min_dte]
        
        if not valid_exps:
            logger.warning("No valid expirations found")
            return None, None, 0
        
        # Use first valid expiration
        expiration = valid_exps[0]
        chain = self.data_fetcher.get_chain(symbol, expiration.strftime("%Y-%m-%d"))
        
        # Find put at target delta
        put = self.data_fetcher.get_otm_put(chain, target_delta)
        
        # Calculate cost
        cost = (put.ask + put.bid) / 2 if put.bid > 0 else put.ask
        
        return put, expiration, cost
    
    def buy_protection(
        self,
        symbol: str,
        shares: int,
        underlying_price: Optional[float] = None,
        target_delta: Optional[float] = None
    ) -> ProtectivePutPosition:
        """
        Buy protective puts for a position.
        
        Args:
            symbol: Underlying symbol
            shares: Number of shares to protect
            underlying_price: Current price (auto-fetched if not provided)
            target_delta: Delta target for puts
        """
        if underlying_price is None:
            # Get from chain
            exps = self.data_fetcher.get_expirations(symbol)
            if exps:
                chain = self.data_fetcher.get_chain(symbol, exps[0].strftime("%Y-%m-%d"))
                underlying_price = chain.underlying_price
            else:
                underlying_price = 100.0
        
        put, expiration, cost_per_contract = self.find_optimal_put(
            symbol, underlying_price, target_delta
        )
        
        if put is None:
            logger.error("Could not find suitable put")
            return None
        
        # Calculate contracts needed (1 contract = 100 shares)
        contracts = max(1, shares // 100)
        total_cost = cost_per_contract * contracts * 100
        
        # Protection level
        protection_pct = (underlying_price - put.strike) / underlying_price
        
        position = ProtectivePutPosition(
            underlying=symbol,
            shares=shares,
            put_strike=put.strike,
            put_expiration=expiration,
            put_cost=total_cost,
            delta=target_delta or self.target_delta,
            protection_level=protection_pct
        )
        
        logger.info(f"Protective put: {contracts}x {put.strike}P @ ${cost_per_contract:.2f}")
        logger.info(f"Total cost: ${total_cost:.2f} ({total_cost / (underlying_price * shares) * 100:.2f}%)")
        
        return position
    
    def simulate(
        self,
        position: ProtectivePutPosition,
        underlying_returns: List[float]
    ) -> List[ProtectivePutResult]:
        """
        Simulate protective put payoff across scenarios.
        
        Args:
            position: The protective put position
            underlying_returns: List of return scenarios to simulate
        """
        results = []
        
        # Get initial underlying price (estimate from strike)
        initial_price = position.put_strike / (1 - position.protection_level)
        position_value = initial_price * position.shares
        cost_pct = position.put_cost / position_value
        
        for ret in underlying_returns:
            final_price = initial_price * (1 + ret)
            
            # Underlying P&L
            underlying_pnl = (final_price - initial_price) * position.shares
            underlying_return = ret
            
            # Put payoff (simplified - at expiration)
            if final_price < position.put_strike:
                # Put is ITM
                put_payoff = (position.put_strike - final_price) * position.shares
            else:
                # Put expires worthless
                put_payoff = 0
            
            # Net P&L
            put_pnl = put_payoff - position.put_cost
            total_pnl = underlying_pnl + put_pnl
            protected_return = total_pnl / position_value
            
            # Max loss with protection
            max_loss_price = position.put_strike
            max_loss = ((max_loss_price - initial_price) * position.shares - position.put_cost) / position_value
            
            results.append(ProtectivePutResult(
                underlying_return=underlying_return,
                protected_return=protected_return,
                max_loss=max_loss,
                put_pnl=put_pnl / position_value,
                total_cost_pct=cost_pct
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
        Backtest protective puts during Risk-Off periods.
        
        Args:
            symbol: Underlying symbol
            returns: Historical daily returns
            regime_signals: Regime signals ("RISK_ON", "RISK_OFF")
            shares: Position size
        """
        total_pnl = 0.0
        put_costs = 0.0
        protection_events = 0
        protected_losses = 0.0
        
        position = None
        
        for i, (ret, regime) in enumerate(zip(returns, regime_signals)):
            if regime == "RISK_OFF" and position is None:
                # Enter protection
                position = self.buy_protection(symbol, shares)
                if position:
                    put_costs += position.put_cost
                    protection_events += 1
            
            elif regime == "RISK_ON" and position is not None:
                # Exit protection (let puts expire/sell)
                position = None
            
            # Calculate P&L
            if position is not None and ret < -0.02:  # Significant down day
                # Protection kicks in
                protected_loss = max(ret, -position.protection_level)
                protected_losses += (ret - protected_loss) * shares * 100
            
            total_pnl += ret * shares * 100
        
        return {
            "total_return": total_pnl,
            "put_costs": put_costs,
            "net_return": total_pnl - put_costs,
            "protection_events": protection_events,
            "protected_losses": protected_losses,
            "cost_per_protection": put_costs / max(1, protection_events)
        }


if __name__ == "__main__":
    strategy = ProtectivePutStrategy()
    
    # Buy protection
    position = strategy.buy_protection("SPY", 100)
    
    if position:
        print(f"\nPosition: {position.shares} shares + {position.put_strike}P")
        print(f"Cost: ${position.put_cost:.2f}")
        print(f"Protection level: {position.protection_level*100:.1f}% OTM")
        
        # Simulate scenarios
        scenarios = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10]
        results = strategy.simulate(position, scenarios)
        
        print("\nScenario Analysis:")
        for ret, result in zip(scenarios, results):
            print(f"  Market {ret*100:+.0f}% → Protected {result.protected_return*100:+.1f}%")
