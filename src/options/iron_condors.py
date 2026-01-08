"""
Iron Condor Strategy

During low-vol + confirmed Trend regimes only:
- Sell put spread + call spread for premium
- Strict width caps and risk limits

Paper-first. Only after initial experiments succeed.
"""

import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IronCondorPosition:
    """An iron condor position."""
    underlying: str
    short_put_strike: float
    long_put_strike: float
    short_call_strike: float
    long_call_strike: float
    expiration: date
    premium_received: float
    max_loss: float
    width: float  # Width of each spread
    contracts: int


@dataclass
class IronCondorResult:
    """Result of iron condor simulation."""
    underlying_return: float
    pnl: float
    pnl_pct: float
    breached: bool
    breach_side: Optional[str]  # "put" or "call"


class IronCondorStrategy:
    """
    Implements iron condor overlay for low-vol Trend regimes.
    
    ONLY use when:
    - IV is below historical average
    - Regime is confirmed NEUTRAL or FAVORABLE
    - VIX is below 20
    
    Usage:
        strategy = IronCondorStrategy()
        position = strategy.open_condor("SPY", width=5)
        result = strategy.simulate(position, underlying_price_change=0.02)
    """
    
    def __init__(
        self,
        data_fetcher=None,
        max_width: int = 5,  # Max $5 width per spread
        target_delta: float = 0.20,  # 20-delta strikes
        max_position_pct: float = 0.01,  # Max 1% of account per condor
        min_dte: int = 14,
        max_dte: int = 45
    ):
        from .data_fetcher import OptionsDataFetcher
        self.data_fetcher = data_fetcher or OptionsDataFetcher()
        self.max_width = max_width
        self.target_delta = target_delta
        self.max_position_pct = max_position_pct
        self.min_dte = min_dte
        self.max_dte = max_dte
        
        logger.info(f"IronCondorStrategy initialized (delta={target_delta}, width=${max_width})")
    
    def check_regime_eligibility(
        self,
        vix: float,
        regime: str,
        iv_percentile: float
    ) -> tuple:
        """
        Check if conditions are suitable for iron condors.
        
        Returns (eligible, reason).
        """
        if vix > 20:
            return False, f"VIX too high: {vix:.1f} > 20"
        
        if regime not in ["NEUTRAL", "FAVORABLE", "RISK_ON"]:
            return False, f"Regime not suitable: {regime}"
        
        if iv_percentile > 0.50:
            return False, f"IV percentile too high: {iv_percentile:.0%} > 50%"
        
        return True, "Conditions suitable"
    
    def open_condor(
        self,
        symbol: str,
        contracts: int = 1,
        width: Optional[int] = None,
        underlying_price: Optional[float] = None
    ) -> IronCondorPosition:
        """
        Open iron condor position.
        
        Args:
            symbol: Underlying symbol
            contracts: Number of contracts
            width: Width of each spread (default: max_width)
            underlying_price: Current price
        """
        width = width or self.max_width
        
        if underlying_price is None:
            exps = self.data_fetcher.get_expirations(symbol)
            if exps:
                chain = self.data_fetcher.get_chain(symbol, exps[0].strftime("%Y-%m-%d"))
                underlying_price = chain.underlying_price
            else:
                underlying_price = 450.0  # Default for SPY
        
        expirations = self.data_fetcher.get_expirations(symbol)
        today = date.today()
        
        # Find suitable expiration
        valid_exps = [e for e in expirations if self.min_dte <= (e - today).days <= self.max_dte]
        
        if not valid_exps:
            logger.warning("No valid expirations found")
            return None
        
        expiration = valid_exps[0]
        chain = self.data_fetcher.get_chain(symbol, expiration.strftime("%Y-%m-%d"))
        
        # Find strikes (~20-delta)
        short_put_strike = round(underlying_price * 0.95, 0)  # ~5% OTM
        long_put_strike = short_put_strike - width
        short_call_strike = round(underlying_price * 1.05, 0)  # ~5% OTM
        long_call_strike = short_call_strike + width
        
        # Estimate premiums (simplified)
        dte = (expiration - today).days
        time_factor = np.sqrt(dte / 30)
        
        # Put spread premium (credit)
        put_spread_credit = 1.00 * time_factor  # ~$1 credit
        
        # Call spread premium (credit)
        call_spread_credit = 1.00 * time_factor  # ~$1 credit
        
        total_premium = (put_spread_credit + call_spread_credit) * contracts * 100
        max_loss = (width * 100 * contracts) - total_premium
        
        position = IronCondorPosition(
            underlying=symbol,
            short_put_strike=short_put_strike,
            long_put_strike=long_put_strike,
            short_call_strike=short_call_strike,
            long_call_strike=long_call_strike,
            expiration=expiration,
            premium_received=total_premium,
            max_loss=max_loss,
            width=width,
            contracts=contracts
        )
        
        logger.info(f"Iron Condor: {long_put_strike}/{short_put_strike}P - {short_call_strike}/{long_call_strike}C")
        logger.info(f"Premium: ${total_premium:.2f}, Max loss: ${max_loss:.2f}")
        
        return position
    
    def simulate(
        self,
        position: IronCondorPosition,
        underlying_prices: List[float]
    ) -> List[IronCondorResult]:
        """
        Simulate iron condor payoff at expiration.
        """
        results = []
        
        # Estimate initial underlying price from strikes
        initial_price = (position.short_put_strike + position.short_call_strike) / 2
        
        for final_price in underlying_prices:
            ret = (final_price - initial_price) / initial_price
            
            # At expiration P&L
            if final_price <= position.long_put_strike:
                # Max loss on put side
                pnl = -position.max_loss
                breached = True
                breach_side = "put"
            
            elif final_price <= position.short_put_strike:
                # Partial loss on put side
                intrinsic = position.short_put_strike - final_price
                pnl = position.premium_received - (intrinsic * position.contracts * 100)
                breached = True
                breach_side = "put"
            
            elif final_price >= position.long_call_strike:
                # Max loss on call side
                pnl = -position.max_loss
                breached = True
                breach_side = "call"
            
            elif final_price >= position.short_call_strike:
                # Partial loss on call side
                intrinsic = final_price - position.short_call_strike
                pnl = position.premium_received - (intrinsic * position.contracts * 100)
                breached = True
                breach_side = "call"
            
            else:
                # Within profit zone
                pnl = position.premium_received
                breached = False
                breach_side = None
            
            pnl_pct = pnl / (position.max_loss + position.premium_received)
            
            results.append(IronCondorResult(
                underlying_return=ret,
                pnl=pnl,
                pnl_pct=pnl_pct,
                breached=breached,
                breach_side=breach_side
            ))
        
        return results
    
    def calculate_probability_of_profit(
        self,
        position: IronCondorPosition,
        iv: float,
        dte: int
    ) -> float:
        """
        Estimate probability of profit.
        """
        # Simplified using standard deviation
        initial_price = (position.short_put_strike + position.short_call_strike) / 2
        daily_vol = iv / np.sqrt(252)
        expected_move = initial_price * daily_vol * np.sqrt(dte)
        
        # Distance to short strikes
        put_distance = initial_price - position.short_put_strike
        call_distance = position.short_call_strike - initial_price
        
        # Z-scores
        put_z = put_distance / expected_move
        call_z = call_distance / expected_move
        
        # Probability (simplified normal approximation)
        from scipy.stats import norm
        prob_above_put = norm.cdf(put_z)
        prob_below_call = norm.cdf(call_z)
        
        pop = prob_above_put * prob_below_call
        
        return pop


if __name__ == "__main__":
    strategy = IronCondorStrategy()
    
    # Check eligibility
    eligible, reason = strategy.check_regime_eligibility(
        vix=15.0,
        regime="NEUTRAL",
        iv_percentile=0.30
    )
    print(f"Eligible: {eligible} - {reason}")
    
    if eligible:
        # Open condor
        position = strategy.open_condor("SPY", contracts=1)
        
        if position:
            print(f"\nPosition:")
            print(f"  Put spread: {position.long_put_strike}/{position.short_put_strike}")
            print(f"  Call spread: {position.short_call_strike}/{position.long_call_strike}")
            print(f"  Premium: ${position.premium_received:.2f}")
            print(f"  Max loss: ${position.max_loss:.2f}")
            
            # Simulate scenarios
            prices = [420, 430, 440, 450, 460, 470, 480]
            results = strategy.simulate(position, prices)
            
            print("\nScenario Analysis:")
            for price, result in zip(prices, results):
                breach = f"BREACH ({result.breach_side})" if result.breached else "PROFIT"
                print(f"  ${price} → ${result.pnl:+.2f} ({result.pnl_pct*100:+.1f}%) {breach}")
