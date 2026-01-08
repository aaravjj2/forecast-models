"""
Integration tests for Executor Service

Tests:
- Full day end-to-end simulation
- Signal → Order → Audit trail
- Idempotency across API calls
"""

import pytest
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

# Import with proper setup
import os
os.environ["TRADING_MODE"] = "simulation"

from automation.executor_service import app, get_state
from automation.order_manager import OrderManager
from automation.safety_monitor import SafetyMonitor


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def reset_state():
    """Reset application state before each test."""
    from automation import executor_service
    executor_service.app_state = None
    yield
    executor_service.app_state = None


class TestHealthCheck:
    """Tests for health endpoint."""
    
    def test_health_returns_ok(self, client, reset_state):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestSignalIngestion:
    """Tests for signal ingestion endpoint."""
    
    def test_hold_signal_no_order(self, client, reset_state):
        response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "HOLD",
            "confidence": 0.5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "HOLD"
        assert data["order_id"] is None
    
    def test_enter_long_creates_order(self, client, reset_state):
        response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "ENTER_LONG",
            "confidence": 0.8,
            "signal_id": "TEST-001"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "ENTER_LONG"
        assert data["order_id"] is not None
    
    def test_exit_creates_order(self, client, reset_state):
        response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "EXIT_TO_CASH",
            "confidence": 0.9
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "EXIT_TO_CASH"
        assert data["order_id"] is not None
    
    def test_idempotent_signals(self, client, reset_state):
        # Send same signal twice
        signal = {
            "symbol": "SPY",
            "signal": "ENTER_LONG",
            "confidence": 0.8,
            "signal_id": "IDEMPOTENT-001"
        }
        
        response1 = client.post("/signal", json=signal)
        response2 = client.post("/signal", json=signal)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Should return same order ID (idempotent)
        data1 = response1.json()
        data2 = response2.json()
        assert data1["order_id"] == data2["order_id"]


class TestDirectOrders:
    """Tests for direct order placement."""
    
    def test_place_market_order(self, client, reset_state):
        response = client.post("/place_order", json={
            "symbol": "SPY",
            "side": "buy",
            "quantity": 100,
            "order_type": "market"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["client_order_id"] is not None
        assert data["status"] in ["filled", "submitted", "pending"]
    
    def test_place_limit_order(self, client, reset_state):
        response = client.post("/place_order", json={
            "symbol": "SPY",
            "side": "buy",
            "quantity": 100,
            "order_type": "limit",
            "limit_price": 450.0
        })
        
        assert response.status_code == 200


class TestOrderStatus:
    """Tests for order status endpoint."""
    
    def test_get_order_status(self, client, reset_state):
        # First create an order
        order_response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "ENTER_LONG",
            "confidence": 0.8
        })
        order_id = order_response.json()["order_id"]
        
        # Get status
        status_response = client.get(f"/order_status/{order_id}")
        assert status_response.status_code == 200
        
        data = status_response.json()
        assert data["client_order_id"] == order_id
        assert data["symbol"] == "SPY"
        assert data["side"] == "buy"
    
    def test_order_not_found(self, client, reset_state):
        response = client.get("/order_status/nonexistent-order-id")
        assert response.status_code == 404


class TestKillSwitch:
    """Tests for kill switch endpoints."""
    
    def test_trigger_kill_switch(self, client, reset_state):
        response = client.post("/kill_switch?reason=test_reason")
        assert response.status_code == 200
        
        # Verify trading is blocked
        signal_response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "ENTER_LONG",
            "confidence": 0.8
        })
        
        assert signal_response.json()["action"] == "BLOCKED"


class TestEndToEndDay:
    """End-to-end test simulating a full trading day."""
    
    def test_full_day_simulation(self, client, reset_state):
        # Morning: Enter long
        enter_response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "ENTER_LONG",
            "confidence": 0.8,
            "regime_state": "NEUTRAL"
        })
        assert enter_response.status_code == 200
        enter_order_id = enter_response.json()["order_id"]
        
        # Verify order status
        status = client.get(f"/order_status/{enter_order_id}")
        assert status.status_code == 200
        
        # End of day: Exit
        exit_response = client.post("/signal", json={
            "symbol": "SPY",
            "signal": "EXIT_TO_CASH",
            "confidence": 0.9,
            "regime_state": "HOSTILE"
        })
        assert exit_response.status_code == 200
        exit_order_id = exit_response.json()["order_id"]
        
        # Both orders should exist
        assert enter_order_id != exit_order_id
        
        # Check health
        health = client.get("/health")
        data = health.json()
        assert data["orders_placed"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
