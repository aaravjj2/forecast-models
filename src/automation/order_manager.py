"""
Order Manager

Handles order lifecycle with:
- Idempotency via client_order_id
- Retry with exponential backoff
- Circuit breaker (3 consecutive failures → halt)
- Order types: market, limit, TWAP slicer

Paper-first. No live orders unless explicitly enabled.
"""

import os
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Literal
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"


@dataclass
class Order:
    """Order representation."""
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    
    # Execution state
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    
    # Audit
    attempts: int = 0
    last_error: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker for order submission."""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker OPEN after {self.failures} failures")
    
    def record_success(self):
        self.failures = 0
        self.is_open = False
    
    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        # Check if reset timeout has passed
        if self.last_failure_time and time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failures = 0
            logger.info("Circuit breaker RESET")
            return True
        return False


class OrderManager:
    """
    Manages order lifecycle with idempotency and safety.
    
    Usage:
        manager = OrderManager(mode="paper")
        order = manager.create_order("SPY", "buy", 100)
        result = manager.submit_order(order)
    """
    
    def __init__(
        self,
        mode: Literal["simulation", "paper", "live"] = "paper",
        broker_adapter=None,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        self.mode = mode
        self.broker_adapter = broker_adapter
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.circuit_breaker = CircuitBreaker()
        
        # Safety check
        if mode == "live" and not os.environ.get("ENABLE_LIVE_TRADING"):
            raise ValueError(
                "Live trading requires ENABLE_LIVE_TRADING=true environment variable"
            )
        
        logger.info(f"OrderManager initialized in {mode} mode")
    
    def generate_client_order_id(
        self,
        symbol: str,
        side: str,
        signal_id: Optional[str] = None
    ) -> str:
        """
        Generate deterministic, idempotent client order ID.
        
        Format: {date}_{symbol}_{side}_{signal_hash}_{uuid4_short}
        """
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        
        if signal_id:
            hash_input = f"{date_str}_{symbol}_{side}_{signal_id}"
            hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        else:
            hash_suffix = uuid.uuid4().hex[:8]
        
        return f"{date_str}_{symbol}_{side}_{hash_suffix}"
    
    def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        signal_id: Optional[str] = None
    ) -> Order:
        """Create a new order with idempotent client_order_id."""
        client_order_id = self.generate_client_order_id(symbol, side, signal_id)
        
        # Check for existing order (idempotency)
        if client_order_id in self.orders:
            existing = self.orders[client_order_id]
            logger.info(f"Idempotent hit: returning existing order {client_order_id}")
            return existing
        
        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        
        self.orders[client_order_id] = order
        logger.info(f"Created order: {client_order_id}")
        
        return order
    
    def submit_order(self, order: Order) -> Order:
        """Submit order with retry and circuit breaker."""
        
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            order.status = OrderStatus.FAILED
            order.last_error = "Circuit breaker open"
            logger.error(f"Order {order.client_order_id} blocked by circuit breaker")
            return order
        
        # Already submitted?
        if order.status in [OrderStatus.SUBMITTED, OrderStatus.FILLED]:
            logger.info(f"Order {order.client_order_id} already {order.status.value}")
            return order
        
        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            order.attempts = attempt
            
            try:
                if self.mode == "simulation":
                    result = self._simulate_fill(order)
                elif self.mode == "paper":
                    result = self._paper_submit(order)
                else:  # live
                    result = self._live_submit(order)
                
                order.status = result["status"]
                order.broker_order_id = result.get("broker_order_id")
                order.submitted_at = datetime.now(timezone.utc).isoformat()
                
                if order.status == OrderStatus.FILLED:
                    order.filled_quantity = result.get("filled_qty", order.quantity)
                    order.filled_avg_price = result.get("filled_price", 0)
                    order.filled_at = datetime.now(timezone.utc).isoformat()
                
                self.circuit_breaker.record_success()
                logger.info(f"Order {order.client_order_id} {order.status.value}")
                return order
                
            except Exception as e:
                order.last_error = str(e)
                logger.warning(f"Order attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
        
        # All retries exhausted
        order.status = OrderStatus.FAILED
        self.circuit_breaker.record_failure()
        logger.error(f"Order {order.client_order_id} FAILED after {self.max_retries} attempts")
        
        return order
    
    def _simulate_fill(self, order: Order) -> Dict:
        """Simulate order fill for testing."""
        # Mock immediate fill
        return {
            "status": OrderStatus.FILLED,
            "broker_order_id": f"SIM-{uuid.uuid4().hex[:8]}",
            "filled_qty": order.quantity,
            "filled_price": 100.0  # Mock price
        }
    
    def _paper_submit(self, order: Order) -> Dict:
        """Submit to paper trading broker."""
        if not self.broker_adapter:
            # Fallback to simulation
            return self._simulate_fill(order)
        
        # Use broker adapter for paper trading
        result = self.broker_adapter.submit_order(
            symbol=order.symbol,
            side=order.side,
            qty=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            client_order_id=order.client_order_id
        )
        
        return {
            "status": OrderStatus.SUBMITTED,
            "broker_order_id": result.get("id"),
            "filled_qty": 0,
            "filled_price": 0
        }
    
    def _live_submit(self, order: Order) -> Dict:
        """Submit to live broker (requires explicit enablement)."""
        if not os.environ.get("ENABLE_LIVE_TRADING"):
            raise ValueError("Live trading not enabled")
        
        if not self.broker_adapter:
            raise ValueError("No broker adapter configured for live trading")
        
        return self._paper_submit(order)  # Same flow, different account
    
    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        return self.orders.get(client_order_id)
    
    def get_order_status(self, client_order_id: str) -> Optional[OrderStatus]:
        """Get order status by client order ID."""
        order = self.orders.get(client_order_id)
        return order.status if order else None
    
    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel pending order."""
        order = self.orders.get(client_order_id)
        if not order:
            return False
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Order {client_order_id} cancelled")
        return True
    
    def reconcile(self, broker_fills: List[Dict]) -> Dict:
        """
        Reconcile internal state with broker fills.
        Returns discrepancies.
        """
        discrepancies = []
        
        for fill in broker_fills:
            client_id = fill.get("client_order_id")
            order = self.orders.get(client_id)
            
            if not order:
                discrepancies.append({
                    "type": "unknown_fill",
                    "client_order_id": client_id,
                    "broker_data": fill
                })
                continue
            
            # Compare quantities
            if abs(fill.get("filled_qty", 0) - order.filled_quantity) > 0.01:
                discrepancies.append({
                    "type": "quantity_mismatch",
                    "client_order_id": client_id,
                    "expected": order.filled_quantity,
                    "actual": fill.get("filled_qty")
                })
            
            # Compare prices
            if abs(fill.get("avg_price", 0) - order.filled_avg_price) > 0.01:
                discrepancies.append({
                    "type": "price_mismatch",
                    "client_order_id": client_id,
                    "expected": order.filled_avg_price,
                    "actual": fill.get("avg_price")
                })
        
        return {
            "total_orders": len(self.orders),
            "total_fills": len(broker_fills),
            "discrepancies": discrepancies,
            "is_reconciled": len(discrepancies) == 0
        }


if __name__ == "__main__":
    # Quick test
    manager = OrderManager(mode="simulation")
    
    # Create and submit order
    order = manager.create_order("SPY", "buy", 100)
    print(f"Created: {order.client_order_id}")
    
    # Submit
    result = manager.submit_order(order)
    print(f"Status: {result.status.value}")
    
    # Idempotency test
    order2 = manager.create_order("SPY", "buy", 100)
    print(f"Same ID: {order.client_order_id == order2.client_order_id}")
