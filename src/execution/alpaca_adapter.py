"""
Alpaca Paper Trading Adapter

Provides a clean interface to Alpaca's Paper Trading API.
Handles authentication, order submission, position tracking, and account state.

Environment Variables Required:
    ALPACA_API_KEY: Your Alpaca API Key
    ALPACA_SECRET_KEY: Your Alpaca Secret Key
    ALPACA_BASE_URL: (Optional) Defaults to paper trading endpoint
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

# Load keys.env
from dotenv import load_dotenv

# Look for keys.env in parent directories
_keys_env_paths = [
    Path(__file__).parent.parent.parent.parent / "keys.env",  # /home/aarav/Forecast models/keys.env
    Path(__file__).parent.parent.parent / "keys.env",  # project root
    Path.home() / "Forecast models" / "keys.env",
]
for _path in _keys_env_paths:
    if _path.exists():
        load_dotenv(_path)
        break

# Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
except ImportError:
    raise ImportError(
        "alpaca-py is required. Install with: pip install alpaca-py"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExposureState(Enum):
    """Current portfolio exposure state."""
    FLAT = "FLAT"
    LONG = "LONG"


@dataclass
class AccountState:
    """Snapshot of account state."""
    cash: float
    equity: float
    buying_power: float
    exposure: ExposureState
    position_qty: float
    position_market_value: float
    position_avg_entry: Optional[float]
    unrealized_pnl: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderResult:
    """Result of an order submission."""
    order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_qty: Optional[float] = None


class AlpacaAdapter:
    """
    Paper trading adapter for Alpaca.
    
    Usage:
        adapter = AlpacaAdapter()
        state = adapter.get_account_state("AAPL")
        result = adapter.submit_market_order("AAPL", "buy", 100)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca adapter.
        
        Args:
            api_key: Alpaca API key (or set APCA_API_KEY_ID env var)
            secret_key: Alpaca secret key (or set APCA_API_SECRET_KEY env var)
            base_url: API base URL (defaults to paper trading)
            paper: Use paper trading (default True)
        """
        # Try multiple env var names for flexibility
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials required. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                "in keys.env or pass them directly."
            )
        
        self.paper = paper
        self.base_url = base_url or os.environ.get(
            "APCA_ENDPOINT",
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        )
        
        # Initialize clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        logger.info(f"AlpacaAdapter initialized (paper={paper})")
    
    def get_account_state(self, symbol: str) -> AccountState:
        """
        Get current account and position state.
        
        Args:
            symbol: Stock symbol to check position for
            
        Returns:
            AccountState with current portfolio snapshot
        """
        # Get account info
        account = self.trading_client.get_account()
        
        # Get position (if any)
        try:
            position = self.trading_client.get_open_position(symbol)
            exposure = ExposureState.LONG if float(position.qty) > 0 else ExposureState.FLAT
            position_qty = float(position.qty)
            position_mv = float(position.market_value)
            position_avg = float(position.avg_entry_price)
            unrealized_pnl = float(position.unrealized_pl)
        except Exception:
            # No position
            exposure = ExposureState.FLAT
            position_qty = 0.0
            position_mv = 0.0
            position_avg = None
            unrealized_pnl = 0.0
        
        return AccountState(
            cash=float(account.cash),
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            exposure=exposure,
            position_qty=position_qty,
            position_market_value=position_mv,
            position_avg_entry=position_avg,
            unrealized_pnl=unrealized_pnl
        )
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest quote price for a symbol."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.data_client.get_stock_latest_quote(request)
        quote = quotes[symbol]
        # Use midpoint of bid/ask
        return (quote.bid_price + quote.ask_price) / 2
    
    def submit_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        time_in_force: str = "day"
    ) -> OrderResult:
        """
        Submit a market order.
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            qty: Number of shares (mutually exclusive with notional)
            notional: Dollar amount (mutually exclusive with qty)
            time_in_force: "day", "gtc", "opg" (at open), "cls" (at close)
            
        Returns:
            OrderResult with order details
        """
        if qty is None and notional is None:
            raise ValueError("Must specify either qty or notional")
        
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "opg": TimeInForce.OPG,  # At market open
            "cls": TimeInForce.CLS,  # At market close
        }
        
        if notional is not None:
            notional = round(notional, 2)
            
        order_data = MarketOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            qty=qty,
            notional=notional,
            time_in_force=tif_map.get(time_in_force, TimeInForce.DAY)
        )
        
        logger.info(f"Submitting {side} order: {symbol} qty={qty} notional={notional}")
        
        order = self.trading_client.submit_order(order_data)
        
        return OrderResult(
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side.value,
            qty=float(order.qty) if order.qty else 0,
            status=order.status.value,
            submitted_at=order.submitted_at
        )
    
    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """
        Close all positions in a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            OrderResult if position existed, None otherwise
        """
        try:
            order = self.trading_client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return OrderResult(
                order_id=str(order.id),
                symbol=order.symbol,
                side=order.side.value,
                qty=float(order.qty) if order.qty else 0,
                status=order.status.value,
                submitted_at=order.submitted_at
            )
        except Exception as e:
            logger.warning(f"No position to close for {symbol}: {e}")
            return None
    
    def get_order_status(self, order_id: str) -> OrderResult:
        """Get status of an order by ID."""
        order = self.trading_client.get_order_by_id(order_id)
        
        return OrderResult(
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side.value,
            qty=float(order.qty) if order.qty else 0,
            status=order.status.value,
            submitted_at=order.submitted_at,
            filled_at=order.filled_at,
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            filled_qty=float(order.filled_qty) if order.filled_qty else None
        )
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        cancelled = self.trading_client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        logger.info(f"Cancelled {count} orders")
        return count
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open


if __name__ == "__main__":
    # Quick test (requires credentials)
    try:
        adapter = AlpacaAdapter()
        print("Market open:", adapter.is_market_open())
        
        state = adapter.get_account_state("AAPL")
        print(f"Account State: ${state.equity:.2f} equity, {state.exposure.value}")
        
        price = adapter.get_latest_price("AAPL")
        print(f"AAPL Price: ${price:.2f}")
        
    except ValueError as e:
        print(f"Setup required: {e}")
