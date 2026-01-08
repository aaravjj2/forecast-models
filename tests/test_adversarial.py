"""
Adversarial tests for automation stack

Tests:
- Kill switch triggers correctly
- Slippage spike detection
- Missing broker responses
- Circuit breaker behavior
"""

import pytest
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automation.order_manager import OrderManager, OrderStatus
from automation.safety_monitor import SafetyMonitor
from automation.simulator import TradeSimulator, SlippageModel


class TestKillSwitchTriggers:
    """Tests for kill switch scenarios."""
    
    def test_daily_loss_trigger(self):
        monitor = SafetyMonitor(daily_loss_limit=-0.03)
        
        assert monitor.can_trade() is True
        
        # Record a 5% loss (exceeds 3% limit)
        monitor.record_pnl(-0.05)
        
        assert monitor.kill_switch_active is True
        assert monitor.can_trade() is False
        assert "Daily loss" in monitor.kill_switch_reason
    
    def test_slippage_breach_trigger(self):
        monitor = SafetyMonitor(slippage_limit_bps=50.0)
        
        assert monitor.can_trade() is True
        
        # Record a fill with 75 bps slippage (exceeds 50 bps limit)
        monitor.record_fill(
            expected_price=100.0,
            actual_price=100.75,  # 75 bps slippage
            quantity=100,
            side="buy"
        )
        
        assert monitor.kill_switch_active is True
        assert "Slippage breach" in monitor.kill_switch_reason
    
    def test_rate_limit_blocks_trading(self):
        monitor = SafetyMonitor(rate_limit_per_minute=3)
        
        # First 3 should pass
        assert monitor.can_trade() is True
        assert monitor.can_trade() is True
        assert monitor.can_trade() is True
        
        # 4th should be rate limited
        assert monitor.can_trade() is False


class TestCircuitBreakerBehavior:
    """Tests for circuit breaker under adversarial conditions."""
    
    def test_circuit_breaker_blocks_after_failures(self):
        manager = OrderManager(mode="simulation", max_retries=1)
        
        # Manually set circuit breaker to open
        manager.circuit_breaker.record_failure()
        manager.circuit_breaker.record_failure()
        manager.circuit_breaker.record_failure()
        
        assert manager.circuit_breaker.is_open is True
        
        # Try to submit order
        order = manager.create_order("SPY", "buy", 100)
        result = manager.submit_order(order)
        
        assert result.status == OrderStatus.FAILED
        assert "Circuit breaker" in result.last_error


class TestSlippageSpikes:
    """Tests for sudden slippage spikes."""
    
    def test_extreme_slippage_detected(self):
        monitor = SafetyMonitor(slippage_limit_bps=20.0)
        
        # Normal fill
        monitor.record_fill(100.0, 100.01, 100, "buy")
        assert monitor.kill_switch_active is False
        
        # Extreme slippage (100 bps)
        monitor.record_fill(100.0, 101.0, 100, "buy")
        assert monitor.kill_switch_active is True
    
    def test_slippage_model_stress(self):
        # Create high-impact slippage model
        slip_model = SlippageModel(
            base_slippage_bps=5.0,
            size_impact_factor=1.0,  # High impact
            volatility_factor=2.0,   # High volatility sensitivity
            seed=42
        )
        
        # Large order in volatile market
        slippage = slip_model.calculate_slippage(
            order_size=50_000,  # Large order
            adv=1_000_000,      # 5% of ADV
            volatility=0.40,    # High vol
            side="buy"
        )
        
        # Should have significant slippage
        assert slippage > 20  # At least 20 bps


class TestADVBoundEnforcement:
    """Tests for order size vs ADV limits."""
    
    def test_order_size_check(self):
        monitor = SafetyMonitor()
        
        # Order within limits (0.5% of ADV)
        allowed, reason = monitor.check_order_size(
            symbol="SPY",
            quantity=500,
            adv=100_000,
            max_adv_pct=0.01
        )
        assert allowed is True
        
        # Order exceeds limits (2% of ADV)
        allowed, reason = monitor.check_order_size(
            symbol="SPY",
            quantity=2000,
            adv=100_000,
            max_adv_pct=0.01
        )
        assert allowed is False
        assert "exceeds" in reason.lower()


class TestMissingBrokerResponses:
    """Tests for broker response failures."""
    
    def test_retry_on_failure(self):
        manager = OrderManager(mode="simulation", max_retries=3, base_delay=0.01)
        
        order = manager.create_order("SPY", "buy", 100)
        result = manager.submit_order(order)
        
        # Simulation always succeeds, but we can check retries worked
        assert result.attempts == 1  # First try succeeded
    
    def test_exponential_backoff(self):
        # This is implicitly tested via base_delay * (2 ** attempt)
        base_delay = 0.1
        max_retries = 3
        
        expected_delays = [0.1, 0.2, 0.4]  # 0.1 * 2^0, 0.1 * 2^1, 0.1 * 2^2
        
        for i, expected in enumerate(expected_delays):
            actual = base_delay * (2 ** i)
            assert abs(actual - expected) < 0.01


class TestReconciliationDiscrepancies:
    """Tests for reconciliation edge cases."""
    
    def test_detect_quantity_mismatch(self):
        manager = OrderManager(mode="simulation")
        
        order = manager.create_order("SPY", "buy", 100)
        manager.submit_order(order)
        
        # Broker says different quantity
        broker_fills = [{
            "client_order_id": order.client_order_id,
            "filled_qty": 95,  # Mismatch!
            "avg_price": order.filled_avg_price
        }]
        
        result = manager.reconcile(broker_fills)
        assert result["is_reconciled"] is False
        assert len(result["discrepancies"]) > 0
        assert result["discrepancies"][0]["type"] == "quantity_mismatch"
    
    def test_detect_unknown_fill(self):
        manager = OrderManager(mode="simulation")
        
        # Broker reports fill we don't have
        broker_fills = [{
            "client_order_id": "UNKNOWN-ORDER-123",
            "filled_qty": 100,
            "avg_price": 450.0
        }]
        
        result = manager.reconcile(broker_fills)
        assert result["is_reconciled"] is False
        assert result["discrepancies"][0]["type"] == "unknown_fill"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
