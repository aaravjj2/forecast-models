"""
Automation package for production trading.

Modules:
- order_manager: Idempotent order lifecycle management
- executor_service: FastAPI REST endpoints
- safety_monitor: Kill switches and safety limits
- queue_worker: Queue-based signal processing
- bot_worker: Daily scheduled execution
- simulator: Trade simulation with slippage
- metrics: Prometheus metrics export
"""

from .order_manager import OrderManager, Order, OrderType, OrderStatus
from .safety_monitor import SafetyMonitor
from .queue_worker import QueueWorker, SignalQueue, QueuedSignal, AuditDatabase
from .bot_worker import BotWorker
from .simulator import TradeSimulator, SlippageModel, ExecutionModeManager

__all__ = [
    'OrderManager',
    'Order',
    'OrderType',
    'OrderStatus',
    'SafetyMonitor',
    'QueueWorker',
    'SignalQueue',
    'QueuedSignal',
    'AuditDatabase',
    'BotWorker',
    'TradeSimulator',
    'SlippageModel',
    'ExecutionModeManager'
]
