"""
Order Manager

Manages the order lifecycle: submission, tracking, fill confirmation.
Provides higher-level trading operations on top of AlpacaAdapter.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, asdict
import json
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.alpaca_adapter import AlpacaAdapter, ExposureState, OrderResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Complete record of a trade for audit trail."""
    timestamp: str
    symbol: str
    action: str  # ENTER_LONG, EXIT_TO_CASH, HOLD
    signal_source: str  # regime_model, kill_switch, manual
    expected_price: Optional[float]
    order_id: Optional[str]
    order_status: Optional[str]
    filled_price: Optional[float]
    filled_qty: Optional[float]
    slippage_bps: Optional[float]
    notes: str = ""


class OrderManager:
    """
    High-level order management.
    
    Handles:
    - Trade decision execution (enter/exit based on signals)
    - Fill tracking and slippage calculation
    - Trade audit logging
    """
    
    def __init__(
        self,
        symbol: str = "SPY",
        log_dir: Optional[Path] = None,
        adapter: Optional[AlpacaAdapter] = None
    ):
        """
        Initialize OrderManager.
        
        Args:
            symbol: Primary trading symbol
            log_dir: Directory for trade logs
            adapter: AlpacaAdapter instance (creates new if None)
        """
        self.symbol = symbol
        self.log_dir = log_dir or Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.adapter = adapter or AlpacaAdapter()
        
        # Trade log file
        self.trade_log_path = self.log_dir / "paper_trades.csv"
        self._init_trade_log()
        
        logger.info(f"OrderManager initialized for {symbol}")
    
    def _init_trade_log(self):
        """Initialize trade log CSV with headers if it doesn't exist."""
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(TradeRecord.__annotations__.keys()))
                writer.writeheader()
    
    def _log_trade(self, record: TradeRecord):
        """Append trade record to CSV log."""
        with open(self.trade_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(TradeRecord.__annotations__.keys()))
            writer.writerow(asdict(record))
        logger.info(f"Trade logged: {record.action} {record.symbol}")
    
    def get_current_exposure(self) -> ExposureState:
        """Get current portfolio exposure state."""
        state = self.adapter.get_account_state(self.symbol)
        return state.exposure
    
    def execute_signal(
        self,
        signal: Literal["ENTER_LONG", "EXIT_TO_CASH", "HOLD"],
        signal_source: str = "regime_model",
        position_pct: float = 1.0,
        notes: str = ""
    ) -> Optional[TradeRecord]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal (ENTER_LONG, EXIT_TO_CASH, HOLD)
            signal_source: Origin of signal (regime_model, kill_switch, manual)
            position_pct: Fraction of buying power to use (0.0-1.0)
            notes: Optional notes for audit trail
            
        Returns:
            TradeRecord if order was placed, None if HOLD or no action needed
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        state = self.adapter.get_account_state(self.symbol)
        
        # Get expected price for slippage calculation
        try:
            expected_price = self.adapter.get_latest_price(self.symbol)
        except Exception:
            expected_price = None
        
        # HOLD - No action
        if signal == "HOLD":
            record = TradeRecord(
                timestamp=timestamp,
                symbol=self.symbol,
                action="HOLD",
                signal_source=signal_source,
                expected_price=expected_price,
                order_id=None,
                order_status=None,
                filled_price=None,
                filled_qty=None,
                slippage_bps=None,
                notes=f"Current: {state.exposure.value}. {notes}"
            )
            self._log_trade(record)
            return record
        
        # EXIT_TO_CASH - Close position if long
        if signal == "EXIT_TO_CASH":
            if state.exposure == ExposureState.FLAT:
                record = TradeRecord(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action="EXIT_TO_CASH",
                    signal_source=signal_source,
                    expected_price=expected_price,
                    order_id=None,
                    order_status="NO_ACTION",
                    filled_price=None,
                    filled_qty=None,
                    slippage_bps=None,
                    notes=f"Already flat. {notes}"
                )
                self._log_trade(record)
                return record
            
            # Close position
            result = self.adapter.close_position(self.symbol)
            if result:
                record = TradeRecord(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action="EXIT_TO_CASH",
                    signal_source=signal_source,
                    expected_price=expected_price,
                    order_id=result.order_id,
                    order_status=result.status,
                    filled_price=result.filled_price,
                    filled_qty=result.filled_qty,
                    slippage_bps=self._calc_slippage(expected_price, result.filled_price),
                    notes=notes
                )
            else:
                record = TradeRecord(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action="EXIT_TO_CASH",
                    signal_source=signal_source,
                    expected_price=expected_price,
                    order_id=None,
                    order_status="FAILED",
                    filled_price=None,
                    filled_qty=None,
                    slippage_bps=None,
                    notes=f"Close failed. {notes}"
                )
            self._log_trade(record)
            return record
        
        # ENTER_LONG - Buy if not already long
        if signal == "ENTER_LONG":
            if state.exposure == ExposureState.LONG:
                record = TradeRecord(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action="ENTER_LONG",
                    signal_source=signal_source,
                    expected_price=expected_price,
                    order_id=None,
                    order_status="NO_ACTION",
                    filled_price=None,
                    filled_qty=None,
                    slippage_bps=None,
                    notes=f"Already long ({state.position_qty} shares). {notes}"
                )
                self._log_trade(record)
                return record
            
            # Calculate order size
            available = state.buying_power * position_pct * 0.95  # 5% buffer
            
            if available < 1:
                record = TradeRecord(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    action="ENTER_LONG",
                    signal_source=signal_source,
                    expected_price=expected_price,
                    order_id=None,
                    order_status="INSUFFICIENT_FUNDS",
                    filled_price=None,
                    filled_qty=None,
                    slippage_bps=None,
                    notes=f"Buying power: ${state.buying_power:.2f}. {notes}"
                )
                self._log_trade(record)
                return record
            
            # Submit market order (notional)
            result = self.adapter.submit_market_order(
                symbol=self.symbol,
                side="buy",
                notional=available,
                time_in_force="day"
            )
            
            record = TradeRecord(
                timestamp=timestamp,
                symbol=self.symbol,
                action="ENTER_LONG",
                signal_source=signal_source,
                expected_price=expected_price,
                order_id=result.order_id,
                order_status=result.status,
                filled_price=result.filled_price,
                filled_qty=result.filled_qty,
                slippage_bps=self._calc_slippage(expected_price, result.filled_price),
                notes=f"Notional: ${available:.2f}. {notes}"
            )
            self._log_trade(record)
            return record
        
        return None
    
    def _calc_slippage(
        self,
        expected: Optional[float],
        actual: Optional[float]
    ) -> Optional[float]:
        """Calculate slippage in basis points."""
        if expected is None or actual is None or expected == 0:
            return None
        return ((actual - expected) / expected) * 10000
    
    def update_fill_info(self, order_id: str) -> Optional[TradeRecord]:
        """
        Update trade log with fill information for an order.
        
        Call this after order submission to get final fill price.
        """
        try:
            result = self.adapter.get_order_status(order_id)
            logger.info(
                f"Order {order_id}: status={result.status}, "
                f"filled_price={result.filled_price}, filled_qty={result.filled_qty}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
    
    def emergency_flatten(self, reason: str = "kill_switch") -> TradeRecord:
        """
        Emergency: Close all positions and cancel all orders.
        
        Called by kill switches.
        """
        logger.warning(f"EMERGENCY FLATTEN: {reason}")
        
        # Cancel all pending orders
        self.adapter.cancel_all_orders()
        
        # Close position
        return self.execute_signal(
            signal="EXIT_TO_CASH",
            signal_source=reason,
            notes="EMERGENCY FLATTEN"
        )


if __name__ == "__main__":
    # Quick test
    try:
        om = OrderManager(symbol="SPY")
        print(f"Current exposure: {om.get_current_exposure().value}")
        
        # Test HOLD signal
        record = om.execute_signal("HOLD", signal_source="test")
        print(f"HOLD result: {record}")
        
    except Exception as e:
        print(f"Setup required: {e}")
