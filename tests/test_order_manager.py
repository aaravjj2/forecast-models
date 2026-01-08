"""
Unit tests for Order Manager

Tests:
- Order creation
- Idempotency
- Retry with backoff
- Circuit breaker
- Reconciliation
"""

import pytest
import time
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automation.order_manager import (
    OrderManager, Order, OrderType, OrderStatus, CircuitBreaker
)


class TestCircuitBreaker:
    """Tests for circuit breaker."""
    
    def test_initial_state(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.can_proceed() is True
        assert cb.is_open is False
    
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        assert cb.can_proceed() is True
        
        cb.record_failure()
        assert cb.can_proceed() is True
        
        cb.record_failure()  # 3rd failure
        assert cb.is_open is True
        assert cb.can_proceed() is False
    
    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb.failures == 0
        assert cb.is_open is False


class TestOrderManager:
    """Tests for order manager."""
    
    def test_create_order(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        
        assert order.symbol == "SPY"
        assert order.side == "buy"
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
    
    def test_idempotency_same_signal_id(self):
        manager = OrderManager(mode="simulation")
        
        order1 = manager.create_order("SPY", "buy", 100, signal_id="TEST-001")
        order2 = manager.create_order("SPY", "buy", 100, signal_id="TEST-001")
        
        assert order1.client_order_id == order2.client_order_id
        assert len(manager.orders) == 1
    
    def test_different_orders_different_ids(self):
        manager = OrderManager(mode="simulation")
        
        order1 = manager.create_order("SPY", "buy", 100)
        order2 = manager.create_order("GLD", "buy", 50)
        
        assert order1.client_order_id != order2.client_order_id
        assert len(manager.orders) == 2
    
    def test_submit_simulation(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        result = manager.submit_order(order)
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
    
    def test_submit_idempotent(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        
        result1 = manager.submit_order(order)
        result2 = manager.submit_order(order)  # Same order
        
        assert result1.status == OrderStatus.FILLED
        assert result2.status == OrderStatus.FILLED
        assert result1.broker_order_id == result2.broker_order_id
    
    def test_get_order(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        
        retrieved = manager.get_order(order.client_order_id)
        assert retrieved is not None
        assert retrieved.symbol == "SPY"
        
        not_found = manager.get_order("nonexistent")
        assert not_found is None
    
    def test_cancel_order(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        
        success = manager.cancel_order(order.client_order_id)
        assert success is True
        assert order.status == OrderStatus.CANCELLED
    
    def test_reconcile_no_discrepancies(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        manager.submit_order(order)
        
        broker_fills = [{
            "client_order_id": order.client_order_id,
            "filled_qty": order.filled_quantity,
            "avg_price": order.filled_avg_price
        }]
        
        result = manager.reconcile(broker_fills)
        assert result["is_reconciled"] is True
        assert len(result["discrepancies"]) == 0


class TestOrderTypes:
    """Tests for different order types."""
    
    def test_market_order(self):
        manager = OrderManager(mode="simulation")
        order = manager.create_order("SPY", "buy", 100, order_type=OrderType.MARKET)
        assert order.order_type == OrderType.MARKET
    
    def test_limit_order(self):
        manager = OrderManager(mode="simulation")
        order = manager.create_order(
            "SPY", "buy", 100, 
            order_type=OrderType.LIMIT,
            limit_price=450.0
        )
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 450.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
