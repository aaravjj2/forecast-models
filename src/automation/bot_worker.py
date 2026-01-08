"""
Bot Worker

Daily scheduler for T+1 Open execution:
- Runs at configurable time before market open
- Fetches latest regime signal
- Queries meta-decision for position sizing
- Submits idempotent orders via Order Manager

Paper-first. No live orders unless explicitly enabled.
"""

import os
import sys
import logging
import time
import schedule
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotWorker:
    """
    Bot that runs daily to execute the strategy.
    
    Usage:
        bot = BotWorker(symbol="SPY", mode="paper")
        bot.run_schedule()  # Run continuously on schedule
        bot.run_once()     # Run one iteration now
    """
    
    def __init__(
        self,
        symbol: str = "SPY",
        mode: str = "paper",
        execution_time: str = "09:25",  # 5 min before market open
        order_manager=None,
        safety_monitor=None,
        signal_generator=None
    ):
        self.symbol = symbol
        self.mode = mode
        self.execution_time = execution_time
        
        # Lazy imports
        from .order_manager import OrderManager
        from .safety_monitor import SafetyMonitor
        from .queue_worker import AuditDatabase
        
        self.order_manager = order_manager or OrderManager(mode=mode)
        self.safety_monitor = safety_monitor or SafetyMonitor()
        self.signal_generator = signal_generator
        self.audit_db = AuditDatabase()
        
        # State
        self.last_run_date: Optional[str] = None
        self.consecutive_failures = 0
        
        logger.info(f"BotWorker initialized for {symbol} in {mode} mode")
        logger.info(f"Scheduled execution time: {execution_time}")
    
    def _get_signal(self) -> dict:
        """Get latest regime signal."""
        if self.signal_generator:
            return self.signal_generator.generate_signal(self.symbol)
        
        # Fallback: mock signal
        logger.warning("No signal generator configured, using mock signal")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "regime_state": "UNKNOWN",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_position_size(self, signal: dict) -> float:
        """
        Query meta-decision for position size.
        
        Implements conservative sizing rules.
        """
        # Base size
        base_size = 100  # shares
        
        # Scale by confidence
        confidence = signal.get("confidence", 0.5)
        confidence_multiplier = min(1.0, confidence / 0.7)
        
        # Apply regime adjustment
        regime = signal.get("regime_state", "NEUTRAL")
        if regime == "HOSTILE":
            regime_multiplier = 0.5
        elif regime == "FAVORABLE":
            regime_multiplier = 1.0
        else:
            regime_multiplier = 0.75
        
        return int(base_size * confidence_multiplier * regime_multiplier)
    
    def _check_already_run(self) -> bool:
        """Check if already run today (idempotency)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.last_run_date == today
    
    def run_once(self) -> dict:
        """
        Execute one iteration of the strategy.
        Returns execution result.
        """
        logger.info("=" * 60)
        logger.info(f"Bot Worker: Starting execution for {self.symbol}")
        logger.info("=" * 60)
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "action": None,
            "order_id": None,
            "status": None,
            "error": None
        }
        
        try:
            # Idempotency check
            if self._check_already_run():
                logger.info("Already executed today - skipping (idempotent)")
                result["action"] = "SKIP"
                result["status"] = "already_run"
                return result
            
            # Safety check
            if not self.safety_monitor.can_trade():
                reason = self.safety_monitor.get_block_reason()
                logger.warning(f"Trading blocked: {reason}")
                result["action"] = "BLOCKED"
                result["status"] = "safety_blocked"
                result["error"] = reason
                return result
            
            # Get signal
            signal = self._get_signal()
            logger.info(f"Signal: {signal}")
            
            if signal["signal"] == "HOLD":
                logger.info("Signal is HOLD - no action")
                result["action"] = "HOLD"
                result["status"] = "success"
                self.last_run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                return result
            
            # Get position size
            quantity = self._get_position_size(signal)
            logger.info(f"Position size: {quantity}")
            
            if quantity <= 0:
                logger.warning("Position size is 0 - no action")
                result["action"] = "SKIP"
                result["status"] = "zero_size"
                return result
            
            # Determine side
            side = "buy" if signal["signal"] == "ENTER_LONG" else "sell"
            
            # Create and submit order
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side=side,
                quantity=quantity,
                signal_id=f"BOT-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            )
            
            submitted = self.order_manager.submit_order(order)
            
            # Log decision
            self.audit_db.log_decision(
                decision_id=f"DEC-{order.client_order_id}",
                signal_id=signal.get("timestamp", ""),
                action=signal["signal"],
                reasoning=f"Regime: {signal.get('regime_state')}, Confidence: {signal.get('confidence')}",
                position_size=quantity,
                risk_score=1 - signal.get("confidence", 0.5)
            )
            
            result["action"] = signal["signal"]
            result["order_id"] = order.client_order_id
            result["status"] = submitted.status.value
            
            self.last_run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.consecutive_failures = 0
            
            logger.info(f"Execution complete: {result}")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.consecutive_failures += 1
            result["action"] = "ERROR"
            result["status"] = "failed"
            result["error"] = str(e)
            
            if self.consecutive_failures >= 3:
                self.safety_monitor.trigger_kill_switch(
                    f"3 consecutive failures: {e}"
                )
        
        return result
    
    def run_schedule(self):
        """Run on schedule."""
        schedule.every().day.at(self.execution_time).do(self.run_once)
        
        logger.info(f"Scheduler started. Next run at {self.execution_time}")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_continuous(self, interval_seconds: int = 60):
        """Run continuously with specified interval."""
        logger.info(f"Running continuously every {interval_seconds}s")
        
        while True:
            self.run_once()
            time.sleep(interval_seconds)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run bot worker")
    parser.add_argument("--symbol", default="SPY", help="Symbol to trade")
    parser.add_argument("--mode", choices=["simulation", "paper", "live"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--time", default="09:25", help="Execution time (HH:MM)")
    
    args = parser.parse_args()
    
    bot = BotWorker(
        symbol=args.symbol,
        mode=args.mode,
        execution_time=args.time
    )
    
    if args.once:
        result = bot.run_once()
        print(f"Result: {result}")
    else:
        bot.run_schedule()


if __name__ == "__main__":
    main()
