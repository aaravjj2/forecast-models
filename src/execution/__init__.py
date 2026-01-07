"""
Execution package for paper trading.

Modules:
- alpaca_adapter: Alpaca API interface
- order_manager: Order lifecycle management
- live_signal_generator: Regime-based signal generation
- fill_tracker: Slippage tracking
- reconciliation: Weekly reconciliation
"""

from .alpaca_adapter import AlpacaAdapter, ExposureState, AccountState, OrderResult
from .order_manager import OrderManager, TradeRecord
from .live_signal_generator import LiveSignalGenerator, SignalResult
from .fill_tracker import FillTracker, FillRecord
from .reconciliation import WeeklyReconciliation, ReconciliationResult

__all__ = [
    'AlpacaAdapter',
    'ExposureState',
    'AccountState', 
    'OrderResult',
    'OrderManager',
    'TradeRecord',
    'LiveSignalGenerator',
    'SignalResult',
    'FillTracker',
    'FillRecord',
    'WeeklyReconciliation',
    'ReconciliationResult'
]
