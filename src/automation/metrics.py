"""
Prometheus Metrics Export

Metrics for monitoring:
- fill_latency_seconds
- slippage_bps
- orders_total
- orders_success_rate
- pnl_daily
"""

import time
import logging
from typing import Optional
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback to mock if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - using mock metrics")


class MockMetric:
    """Mock metric for when prometheus_client is not available."""
    def __init__(self, *args, **kwargs):
        pass
    def labels(self, *args, **kwargs):
        return self
    def inc(self, *args, **kwargs):
        pass
    def observe(self, *args, **kwargs):
        pass
    def set(self, *args, **kwargs):
        pass


if PROMETHEUS_AVAILABLE:
    # Order metrics
    ORDERS_TOTAL = Counter(
        'volatility_gated_orders_total',
        'Total orders submitted',
        ['symbol', 'side', 'status']
    )
    
    ORDERS_LATENCY = Histogram(
        'volatility_gated_order_latency_seconds',
        'Order submission latency',
        ['symbol'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    FILL_LATENCY = Histogram(
        'volatility_gated_fill_latency_seconds',
        'Time from submission to fill',
        ['symbol'],
        buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
    )
    
    SLIPPAGE_BPS = Histogram(
        'volatility_gated_slippage_bps',
        'Slippage in basis points',
        ['symbol', 'side'],
        buckets=[0, 2, 5, 10, 20, 50, 100, 200]
    )
    
    # Position metrics
    POSITION_VALUE = Gauge(
        'volatility_gated_position_value_usd',
        'Current position value',
        ['symbol']
    )
    
    EXPOSURE_PCT = Gauge(
        'volatility_gated_exposure_pct',
        'Portfolio exposure percentage'
    )
    
    # PnL metrics
    DAILY_PNL = Gauge(
        'volatility_gated_daily_pnl_usd',
        'Daily P&L in USD'
    )
    
    DAILY_PNL_PCT = Gauge(
        'volatility_gated_daily_pnl_pct',
        'Daily P&L percentage'
    )
    
    CUMULATIVE_PNL = Gauge(
        'volatility_gated_cumulative_pnl_usd',
        'Cumulative P&L in USD'
    )
    
    # Safety metrics
    KILL_SWITCH_STATUS = Gauge(
        'volatility_gated_kill_switch_active',
        'Kill switch status (1=active, 0=inactive)'
    )
    
    CIRCUIT_BREAKER_STATUS = Gauge(
        'volatility_gated_circuit_breaker_open',
        'Circuit breaker status (1=open, 0=closed)'
    )
    
    # Signal metrics
    SIGNALS_RECEIVED = Counter(
        'volatility_gated_signals_total',
        'Total signals received',
        ['signal_type']
    )
    
    REGIME_STATE = Gauge(
        'volatility_gated_regime_state',
        'Current regime state (0=HOSTILE, 1=NEUTRAL, 2=FAVORABLE)',
        ['symbol']
    )
    
else:
    # Mock metrics
    ORDERS_TOTAL = MockMetric()
    ORDERS_LATENCY = MockMetric()
    FILL_LATENCY = MockMetric()
    SLIPPAGE_BPS = MockMetric()
    POSITION_VALUE = MockMetric()
    EXPOSURE_PCT = MockMetric()
    DAILY_PNL = MockMetric()
    DAILY_PNL_PCT = MockMetric()
    CUMULATIVE_PNL = MockMetric()
    KILL_SWITCH_STATUS = MockMetric()
    CIRCUIT_BREAKER_STATUS = MockMetric()
    SIGNALS_RECEIVED = MockMetric()
    REGIME_STATE = MockMetric()


class MetricsCollector:
    """
    Collects and exports metrics.
    
    Usage:
        collector = MetricsCollector()
        collector.start_server(port=9090)
        
        collector.record_order("SPY", "buy", "filled")
        collector.record_slippage("SPY", "buy", 5.2)
    """
    
    def __init__(self):
        self.server_started = False
        logger.info("MetricsCollector initialized")
    
    def start_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - metrics server not started")
            return
        
        if not self.server_started:
            start_http_server(port)
            self.server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
    
    def record_order(self, symbol: str, side: str, status: str):
        """Record order submission."""
        ORDERS_TOTAL.labels(symbol=symbol, side=side, status=status).inc()
    
    def record_order_latency(self, symbol: str, latency_seconds: float):
        """Record order submission latency."""
        ORDERS_LATENCY.labels(symbol=symbol).observe(latency_seconds)
    
    def record_fill_latency(self, symbol: str, latency_seconds: float):
        """Record fill latency."""
        FILL_LATENCY.labels(symbol=symbol).observe(latency_seconds)
    
    def record_slippage(self, symbol: str, side: str, slippage_bps: float):
        """Record slippage."""
        SLIPPAGE_BPS.labels(symbol=symbol, side=side).observe(slippage_bps)
    
    def set_position(self, symbol: str, value_usd: float):
        """Set position value."""
        POSITION_VALUE.labels(symbol=symbol).set(value_usd)
    
    def set_exposure(self, exposure_pct: float):
        """Set portfolio exposure."""
        EXPOSURE_PCT.set(exposure_pct)
    
    def set_daily_pnl(self, pnl_usd: float, pnl_pct: float):
        """Set daily P&L."""
        DAILY_PNL.set(pnl_usd)
        DAILY_PNL_PCT.set(pnl_pct)
    
    def set_cumulative_pnl(self, pnl_usd: float):
        """Set cumulative P&L."""
        CUMULATIVE_PNL.set(pnl_usd)
    
    def set_kill_switch(self, active: bool):
        """Set kill switch status."""
        KILL_SWITCH_STATUS.set(1 if active else 0)
    
    def set_circuit_breaker(self, open: bool):
        """Set circuit breaker status."""
        CIRCUIT_BREAKER_STATUS.set(1 if open else 0)
    
    def record_signal(self, signal_type: str):
        """Record signal received."""
        SIGNALS_RECEIVED.labels(signal_type=signal_type).inc()
    
    def set_regime(self, symbol: str, regime: str):
        """Set regime state."""
        regime_map = {"HOSTILE": 0, "NEUTRAL": 1, "FAVORABLE": 2}
        value = regime_map.get(regime, 1)
        REGIME_STATE.labels(symbol=symbol).set(value)


# Timer decorator for measuring function duration
def timed_metric(metric, labels_func=None):
    """Decorator to measure function duration."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if labels_func:
                    labels = labels_func(*args, **kwargs)
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Quick test
    collector = MetricsCollector()
    
    # Record some metrics
    collector.record_order("SPY", "buy", "filled")
    collector.record_slippage("SPY", "buy", 5.2)
    collector.set_daily_pnl(150.0, 0.15)
    
    print("Metrics recorded successfully")
