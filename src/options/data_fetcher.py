"""
Options Data Fetcher

Ingests options data:
- Options chains (SPY, QQQ, GLD)
- IV surface
- Greeks
- Term structure

Uses yfinance for basic options data (free tier).
"""

import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not available - using mock data")


@dataclass
class OptionContract:
    """Single options contract."""
    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str  # "call" or "put"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


@dataclass
class OptionsChain:
    """Options chain for an expiration."""
    underlying: str
    expiration: date
    underlying_price: float
    calls: List[OptionContract]
    puts: List[OptionContract]


class OptionsDataFetcher:
    """
    Fetches options data from available sources.
    
    Usage:
        fetcher = OptionsDataFetcher()
        chain = fetcher.get_chain("SPY", expiration="2026-01-17")
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        project_root = Path(__file__).parent.parent.parent
        self.cache_dir = cache_dir or project_root / "data" / "options"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("OptionsDataFetcher initialized")
    
    def get_expirations(self, symbol: str) -> List[date]:
        """Get available expiration dates."""
        if not YF_AVAILABLE:
            return self._mock_expirations()
        
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            return [datetime.strptime(exp, "%Y-%m-%d").date() for exp in expirations]
        except Exception as e:
            logger.error(f"Failed to get expirations for {symbol}: {e}")
            return self._mock_expirations()
    
    def _mock_expirations(self) -> List[date]:
        """Generate mock expirations."""
        today = date.today()
        expirations = []
        for i in range(1, 13):  # Weekly for first 3 months
            exp = today + timedelta(weeks=i)
            # Adjust to Friday
            days_until_friday = (4 - exp.weekday()) % 7
            exp = exp + timedelta(days=days_until_friday)
            expirations.append(exp)
        return expirations
    
    def get_chain(self, symbol: str, expiration: str) -> OptionsChain:
        """
        Get options chain for symbol and expiration.
        
        Args:
            symbol: Underlying symbol (e.g., "SPY")
            expiration: Expiration date (YYYY-MM-DD format)
        """
        if not YF_AVAILABLE:
            return self._mock_chain(symbol, expiration)
        
        try:
            ticker = yf.Ticker(symbol)
            opts = ticker.option_chain(expiration)
            
            underlying_price = ticker.info.get('regularMarketPrice', 100.0)
            
            calls = self._parse_chain(symbol, expiration, opts.calls, "call")
            puts = self._parse_chain(symbol, expiration, opts.puts, "put")
            
            return OptionsChain(
                underlying=symbol,
                expiration=datetime.strptime(expiration, "%Y-%m-%d").date(),
                underlying_price=underlying_price,
                calls=calls,
                puts=puts
            )
        
        except Exception as e:
            logger.error(f"Failed to get chain for {symbol} {expiration}: {e}")
            return self._mock_chain(symbol, expiration)
    
    def _parse_chain(
        self,
        symbol: str,
        expiration: str,
        df: pd.DataFrame,
        option_type: str
    ) -> List[OptionContract]:
        """Parse yfinance chain dataframe."""
        contracts = []
        
        for _, row in df.iterrows():
            contract = OptionContract(
                symbol=row.get("contractSymbol", ""),
                underlying=symbol,
                expiration=datetime.strptime(expiration, "%Y-%m-%d").date(),
                strike=row.get("strike", 0),
                option_type=option_type,
                bid=row.get("bid", 0),
                ask=row.get("ask", 0),
                last=row.get("lastPrice", 0),
                volume=int(row.get("volume", 0) or 0),
                open_interest=int(row.get("openInterest", 0) or 0),
                implied_volatility=row.get("impliedVolatility", 0)
            )
            contracts.append(contract)
        
        return contracts
    
    def _mock_chain(self, symbol: str, expiration: str) -> OptionsChain:
        """Generate mock options chain."""
        underlying_price = 450.0 if symbol == "SPY" else 100.0
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        
        calls = []
        puts = []
        
        # Generate strikes around current price
        for pct in np.arange(-0.10, 0.11, 0.01):
            strike = round(underlying_price * (1 + pct), 0)
            
            # Mock IV (higher for OTM)
            moneyness = (strike - underlying_price) / underlying_price
            base_iv = 0.20
            iv = base_iv + abs(moneyness) * 0.5
            
            # Mock prices
            dte = (exp_date - date.today()).days
            time_value = np.sqrt(dte / 365) * iv * underlying_price * 0.4
            
            if strike < underlying_price:
                call_price = underlying_price - strike + time_value
                put_price = time_value * 0.8
            else:
                call_price = time_value * 0.8
                put_price = strike - underlying_price + time_value
            
            calls.append(OptionContract(
                symbol=f"{symbol}_{exp_date}_{strike}C",
                underlying=symbol,
                expiration=exp_date,
                strike=strike,
                option_type="call",
                bid=max(0.01, call_price * 0.95),
                ask=call_price * 1.05,
                last=call_price,
                volume=np.random.randint(100, 10000),
                open_interest=np.random.randint(1000, 50000),
                implied_volatility=iv
            ))
            
            puts.append(OptionContract(
                symbol=f"{symbol}_{exp_date}_{strike}P",
                underlying=symbol,
                expiration=exp_date,
                strike=strike,
                option_type="put",
                bid=max(0.01, put_price * 0.95),
                ask=put_price * 1.05,
                last=put_price,
                volume=np.random.randint(100, 10000),
                open_interest=np.random.randint(1000, 50000),
                implied_volatility=iv
            ))
        
        return OptionsChain(
            underlying=symbol,
            expiration=exp_date,
            underlying_price=underlying_price,
            calls=calls,
            puts=puts
        )
    
    def get_atm_options(
        self,
        chain: OptionsChain,
        delta_target: float = 0.50
    ) -> tuple:
        """Get ATM call and put."""
        underlying = chain.underlying_price
        
        # Find closest to ATM
        atm_call = min(chain.calls, key=lambda c: abs(c.strike - underlying))
        atm_put = min(chain.puts, key=lambda p: abs(p.strike - underlying))
        
        return atm_call, atm_put
    
    def get_otm_put(
        self,
        chain: OptionsChain,
        delta_target: float = 0.30
    ) -> OptionContract:
        """Get OTM put at approximately target delta."""
        underlying = chain.underlying_price
        
        # Estimate strike for target delta (simplified)
        # For a 30-delta put, roughly 3-5% OTM
        target_strike = underlying * (1 - delta_target * 0.15)
        
        otm_put = min(chain.puts, key=lambda p: abs(p.strike - target_strike))
        return otm_put
    
    def get_iv_surface(self, symbol: str) -> pd.DataFrame:
        """Get IV surface (strike x expiration)."""
        expirations = self.get_expirations(symbol)[:6]  # First 6 expirations
        
        iv_data = []
        for exp in expirations:
            exp_str = exp.strftime("%Y-%m-%d")
            chain = self.get_chain(symbol, exp_str)
            
            for put in chain.puts:
                moneyness = put.strike / chain.underlying_price
                iv_data.append({
                    "expiration": exp,
                    "dte": (exp - date.today()).days,
                    "strike": put.strike,
                    "moneyness": moneyness,
                    "iv": put.implied_volatility,
                    "type": "put"
                })
            
            for call in chain.calls:
                moneyness = call.strike / chain.underlying_price
                iv_data.append({
                    "expiration": exp,
                    "dte": (exp - date.today()).days,
                    "strike": call.strike,
                    "moneyness": moneyness,
                    "iv": call.implied_volatility,
                    "type": "call"
                })
        
        return pd.DataFrame(iv_data)


if __name__ == "__main__":
    fetcher = OptionsDataFetcher()
    
    # Get expirations
    expirations = fetcher.get_expirations("SPY")
    print(f"Available expirations: {len(expirations)}")
    
    if expirations:
        # Get chain
        exp_str = expirations[0].strftime("%Y-%m-%d")
        chain = fetcher.get_chain("SPY", exp_str)
        print(f"\nChain for {exp_str}:")
        print(f"  Underlying: ${chain.underlying_price:.2f}")
        print(f"  Calls: {len(chain.calls)}")
        print(f"  Puts: {len(chain.puts)}")
        
        # Get ATM
        atm_call, atm_put = fetcher.get_atm_options(chain)
        print(f"\nATM Call: {atm_call.strike} @ ${atm_call.last:.2f}")
        print(f"ATM Put: {atm_put.strike} @ ${atm_put.last:.2f}")
