"""
Executor Service (FastAPI)

REST endpoints for signal ingestion and order management:
- POST /signal - Ingest regime signal
- POST /place_order - Submit order
- GET /order_status/{order_id} - Query order state
- POST /reconcile - Compare broker fills

Paper-first. No live orders unless --live flag and ENABLE_LIVE_TRADING=true.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn

from .order_manager import OrderManager, Order, OrderType, OrderStatus
from .safety_monitor import SafetyMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class SignalRequest(BaseModel):
    """Regime signal from strategy."""
    symbol: str
    signal: Literal["ENTER_LONG", "EXIT_TO_CASH", "HOLD"]
    confidence: float = Field(ge=0, le=1)
    signal_id: Optional[str] = None
    regime_state: Optional[str] = None
    timestamp: Optional[str] = None


class SignalResponse(BaseModel):
    """Response to signal ingestion."""
    signal_id: str
    action: str
    order_id: Optional[str] = None
    message: str


class OrderRequest(BaseModel):
    """Order submission request."""
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)
    order_type: str = "market"
    limit_price: Optional[float] = None
    signal_id: Optional[str] = None


class OrderResponse(BaseModel):
    """Order response."""
    client_order_id: str
    status: str
    message: str


class OrderStatusResponse(BaseModel):
    """Order status response."""
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    status: str
    filled_quantity: float
    filled_avg_price: float
    created_at: str
    submitted_at: Optional[str]
    filled_at: Optional[str]


class ReconcileResponse(BaseModel):
    """Reconciliation response."""
    total_orders: int
    total_fills: int
    is_reconciled: bool
    discrepancies: list


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.mode = os.environ.get("TRADING_MODE", "paper")
        self.order_manager = OrderManager(mode=self.mode)
        self.safety_monitor = SafetyMonitor()
        self.signals_received = 0
        self.orders_placed = 0
        
        logger.info(f"ExecutorService initialized in {self.mode} mode")


app_state: Optional[AppState] = None


def get_state() -> AppState:
    """Dependency to get application state."""
    global app_state
    if app_state is None:
        app_state = AppState()
    return app_state


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global app_state
    app_state = AppState()
    logger.info("Executor service started")
    yield
    logger.info("Executor service stopped")


app = FastAPI(
    title="Volatility-Gated Executor",
    description="Order execution service for Volatility-Gated Long Exposure strategy",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    state = get_state()
    return {
        "status": "healthy",
        "mode": state.mode,
        "signals_received": state.signals_received,
        "orders_placed": state.orders_placed,
        "circuit_breaker_open": state.order_manager.circuit_breaker.is_open
    }


@app.post("/signal", response_model=SignalResponse)
async def ingest_signal(signal: SignalRequest, state: AppState = Depends(get_state)):
    """
    Ingest regime signal and decide on action.
    
    If signal is ENTER_LONG or EXIT_TO_CASH, generates order.
    If HOLD, no action taken.
    """
    state.signals_received += 1
    signal_id = signal.signal_id or f"SIG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    
    # Safety check
    if not state.safety_monitor.can_trade():
        return SignalResponse(
            signal_id=signal_id,
            action="BLOCKED",
            message=f"Trading blocked: {state.safety_monitor.get_block_reason()}"
        )
    
    # Determine action
    if signal.signal == "HOLD":
        return SignalResponse(
            signal_id=signal_id,
            action="HOLD",
            message="No action required"
        )
    
    # For ENTER/EXIT, we need position sizing (simplified here)
    # In production, this would query the meta-decision module
    quantity = 100  # Placeholder
    
    if signal.signal == "ENTER_LONG":
        order = state.order_manager.create_order(
            symbol=signal.symbol,
            side="buy",
            quantity=quantity,
            signal_id=signal_id
        )
        result = state.order_manager.submit_order(order)
        state.orders_placed += 1
        
        return SignalResponse(
            signal_id=signal_id,
            action="ENTER_LONG",
            order_id=order.client_order_id,
            message=f"Order {result.status.value}"
        )
    
    elif signal.signal == "EXIT_TO_CASH":
        order = state.order_manager.create_order(
            symbol=signal.symbol,
            side="sell",
            quantity=quantity,
            signal_id=signal_id
        )
        result = state.order_manager.submit_order(order)
        state.orders_placed += 1
        
        return SignalResponse(
            signal_id=signal_id,
            action="EXIT_TO_CASH",
            order_id=order.client_order_id,
            message=f"Order {result.status.value}"
        )
    
    return SignalResponse(
        signal_id=signal_id,
        action="UNKNOWN",
        message=f"Unknown signal: {signal.signal}"
    )


@app.post("/place_order", response_model=OrderResponse)
async def place_order(order_req: OrderRequest, state: AppState = Depends(get_state)):
    """
    Place an order directly (bypass signal logic).
    """
    # Safety check
    if not state.safety_monitor.can_trade():
        raise HTTPException(
            status_code=403,
            detail=f"Trading blocked: {state.safety_monitor.get_block_reason()}"
        )
    
    # ADV check would go here in production
    
    order_type = OrderType.MARKET
    if order_req.order_type.lower() == "limit":
        order_type = OrderType.LIMIT
    elif order_req.order_type.lower() == "twap":
        order_type = OrderType.TWAP
    
    order = state.order_manager.create_order(
        symbol=order_req.symbol,
        side=order_req.side,
        quantity=order_req.quantity,
        order_type=order_type,
        limit_price=order_req.limit_price,
        signal_id=order_req.signal_id
    )
    
    result = state.order_manager.submit_order(order)
    state.orders_placed += 1
    
    return OrderResponse(
        client_order_id=result.client_order_id,
        status=result.status.value,
        message=f"Order {result.status.value}"
    )


@app.get("/order_status/{order_id}", response_model=OrderStatusResponse)
async def get_order_status(order_id: str, state: AppState = Depends(get_state)):
    """
    Get order status by client order ID.
    """
    order = state.order_manager.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return OrderStatusResponse(
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        status=order.status.value,
        filled_quantity=order.filled_quantity,
        filled_avg_price=order.filled_avg_price,
        created_at=order.created_at,
        submitted_at=order.submitted_at,
        filled_at=order.filled_at
    )


@app.post("/reconcile", response_model=ReconcileResponse)
async def reconcile_orders(broker_fills: list, state: AppState = Depends(get_state)):
    """
    Reconcile internal state with broker fills.
    """
    result = state.order_manager.reconcile(broker_fills)
    
    return ReconcileResponse(
        total_orders=result["total_orders"],
        total_fills=result["total_fills"],
        is_reconciled=result["is_reconciled"],
        discrepancies=result["discrepancies"]
    )


@app.post("/kill_switch")
async def trigger_kill_switch(reason: str, state: AppState = Depends(get_state)):
    """
    Manually trigger kill switch.
    """
    state.safety_monitor.trigger_kill_switch(reason)
    return {"status": "kill_switch_triggered", "reason": reason}


@app.post("/reset_kill_switch")
async def reset_kill_switch(state: AppState = Depends(get_state)):
    """
    Reset kill switch (requires manual confirmation in production).
    """
    state.safety_monitor.reset_kill_switch()
    return {"status": "kill_switch_reset"}


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the executor service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run executor service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--live", action="store_true", help="Enable live trading mode")
    
    args = parser.parse_args()
    
    if args.live:
        os.environ["TRADING_MODE"] = "live"
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
