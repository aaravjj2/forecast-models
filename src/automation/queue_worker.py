"""
Queue Worker

Queue-based execution pipeline:
- Consumes signals from queue
- Generates orders via Order Manager
- Stores audit trail to database

Supports Redis or in-memory queue fallback.
"""

import os
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from queue import Queue as MemoryQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueuedSignal:
    """Signal in the queue."""
    signal_id: str
    symbol: str
    signal: str  # ENTER_LONG, EXIT_TO_CASH, HOLD
    confidence: float
    regime_state: str
    timestamp: str
    processed: bool = False
    order_id: Optional[str] = None


class AuditDatabase:
    """SQLite audit trail storage."""
    
    def __init__(self, db_path: Optional[Path] = None):
        project_root = Path(__file__).parent.parent.parent
        self.db_path = db_path or project_root / "data" / "audit.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                symbol TEXT,
                signal TEXT,
                confidence REAL,
                regime_state TEXT,
                timestamp TEXT,
                processed INTEGER DEFAULT 0,
                order_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_order_id TEXT UNIQUE,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                order_type TEXT,
                status TEXT,
                broker_order_id TEXT,
                filled_quantity REAL,
                filled_avg_price REAL,
                created_at TEXT,
                submitted_at TEXT,
                filled_at TEXT,
                signal_id TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT UNIQUE,
                signal_id TEXT,
                action TEXT,
                reasoning TEXT,
                position_size REAL,
                risk_score REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Audit database initialized: {self.db_path}")
    
    def log_signal(self, signal: QueuedSignal):
        """Log a signal to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO signals 
                (signal_id, symbol, signal, confidence, regime_state, timestamp, processed, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.signal_id,
                signal.symbol,
                signal.signal,
                signal.confidence,
                signal.regime_state,
                signal.timestamp,
                1 if signal.processed else 0,
                signal.order_id
            ))
            conn.commit()
        finally:
            conn.close()
    
    def log_order(self, order):
        """Log an order to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO orders
                (client_order_id, symbol, side, quantity, order_type, status,
                 broker_order_id, filled_quantity, filled_avg_price,
                 created_at, submitted_at, filled_at, signal_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.client_order_id,
                order.symbol,
                order.side,
                order.quantity,
                order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                order.status.value if hasattr(order.status, 'value') else str(order.status),
                order.broker_order_id,
                order.filled_quantity,
                order.filled_avg_price,
                order.created_at,
                order.submitted_at,
                order.filled_at,
                None  # Signal ID linkage
            ))
            conn.commit()
        finally:
            conn.close()
    
    def log_decision(self, decision_id: str, signal_id: str, action: str, 
                     reasoning: str, position_size: float, risk_score: float):
        """Log a trading decision."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO decisions
                (decision_id, signal_id, action, reasoning, position_size, risk_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (decision_id, signal_id, action, reasoning, position_size, risk_score))
            conn.commit()
        finally:
            conn.close()


class SignalQueue:
    """
    Signal queue with Redis or in-memory backend.
    """
    
    def __init__(self, use_redis: bool = False, redis_url: str = "redis://localhost:6379"):
        self.use_redis = use_redis
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.queue_name = "volatility_gated_signals"
                logger.info(f"Using Redis queue: {redis_url}")
            except ImportError:
                logger.warning("Redis not installed, falling back to in-memory queue")
                self.use_redis = False
        
        if not self.use_redis:
            self.memory_queue = MemoryQueue()
            logger.info("Using in-memory queue")
    
    def push(self, signal: QueuedSignal):
        """Push signal to queue."""
        if self.use_redis:
            self.redis_client.rpush(self.queue_name, json.dumps(asdict(signal)))
        else:
            self.memory_queue.put(signal)
    
    def pop(self, timeout: int = 1) -> Optional[QueuedSignal]:
        """Pop signal from queue."""
        if self.use_redis:
            result = self.redis_client.blpop(self.queue_name, timeout=timeout)
            if result:
                data = json.loads(result[1])
                return QueuedSignal(**data)
            return None
        else:
            try:
                return self.memory_queue.get(timeout=timeout)
            except:
                return None
    
    def size(self) -> int:
        """Get queue size."""
        if self.use_redis:
            return self.redis_client.llen(self.queue_name)
        else:
            return self.memory_queue.qsize()


class QueueWorker:
    """
    Consumes signals from queue and processes them.
    """
    
    def __init__(
        self,
        order_manager,
        safety_monitor,
        signal_queue: Optional[SignalQueue] = None,
        audit_db: Optional[AuditDatabase] = None
    ):
        self.order_manager = order_manager
        self.safety_monitor = safety_monitor
        self.signal_queue = signal_queue or SignalQueue()
        self.audit_db = audit_db or AuditDatabase()
        
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        logger.info("QueueWorker initialized")
    
    def process_signal(self, signal: QueuedSignal) -> Optional[str]:
        """Process a single signal."""
        logger.info(f"Processing signal: {signal.signal_id}")
        
        # Safety check
        if not self.safety_monitor.can_trade():
            logger.warning(f"Trading blocked: {self.safety_monitor.get_block_reason()}")
            signal.processed = True
            self.audit_db.log_signal(signal)
            return None
        
        if signal.signal == "HOLD":
            signal.processed = True
            self.audit_db.log_signal(signal)
            return None
        
        # Determine position size (simplified)
        quantity = 100  # Would query meta-decision in production
        
        side = "buy" if signal.signal == "ENTER_LONG" else "sell"
        
        order = self.order_manager.create_order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            signal_id=signal.signal_id
        )
        
        result = self.order_manager.submit_order(order)
        
        # Log to audit trail
        signal.processed = True
        signal.order_id = order.client_order_id
        self.audit_db.log_signal(signal)
        self.audit_db.log_order(result)
        
        logger.info(f"Signal {signal.signal_id} → Order {order.client_order_id} ({result.status.value})")
        
        return order.client_order_id
    
    def run_once(self):
        """Process one signal from the queue."""
        signal = self.signal_queue.pop(timeout=1)
        if signal:
            return self.process_signal(signal)
        return None
    
    def run_loop(self):
        """Run processing loop."""
        logger.info("Queue worker loop started")
        
        while self.running:
            try:
                self.run_once()
                time.sleep(0.1)  # Small delay to prevent CPU spinning
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                time.sleep(1)
        
        logger.info("Queue worker loop stopped")
    
    def start(self):
        """Start worker in background thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Queue worker started")
    
    def stop(self):
        """Stop worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Queue worker stopped")


if __name__ == "__main__":
    from .order_manager import OrderManager
    from .safety_monitor import SafetyMonitor
    
    # Quick test
    manager = OrderManager(mode="simulation")
    monitor = SafetyMonitor()
    worker = QueueWorker(manager, monitor)
    
    # Push a test signal
    signal = QueuedSignal(
        signal_id="TEST-001",
        symbol="SPY",
        signal="ENTER_LONG",
        confidence=0.8,
        regime_state="NEUTRAL",
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    worker.signal_queue.push(signal)
    print(f"Queue size: {worker.signal_queue.size()}")
    
    # Process
    order_id = worker.run_once()
    print(f"Order ID: {order_id}")
