"""
Safety Monitor

Independent monitor enforcing safety rules:
- Daily loss limit
- Slippage breach detection
- Broker rate limits
- Order size vs ADV bounding

This runs independently of the order manager.
"""

import os
import logging
from datetime import datetime, timezone, date
from typing import Optional, Dict
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    pnl: float = 0.0
    orders_placed: int = 0
    orders_filled: int = 0
    slippage_total_bps: float = 0.0
    max_slippage_bps: float = 0.0


class SafetyMonitor:
    """
    Monitors trading safety and enforces limits.
    
    Kill switches:
    - Daily loss > threshold
    - Slippage > threshold
    - Manual trigger
    """
    
    def __init__(
        self,
        daily_loss_limit: float = -0.03,  # -3%
        slippage_limit_bps: float = 50.0,  # 50 bps
        rate_limit_per_minute: int = 10
    ):
        self.daily_loss_limit = daily_loss_limit
        self.slippage_limit_bps = slippage_limit_bps
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # State
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        self.daily_stats = self._init_daily_stats()
        
        # Rate limiting
        self.request_timestamps: list = []
        
        logger.info(f"SafetyMonitor initialized: loss_limit={daily_loss_limit*100:.1f}%")
    
    def _init_daily_stats(self) -> DailyStats:
        """Initialize daily stats."""
        return DailyStats(date=date.today())
    
    def _check_date_rollover(self):
        """Reset stats on new day."""
        if self.daily_stats.date != date.today():
            logger.info("New trading day - resetting daily stats")
            self.daily_stats = self._init_daily_stats()
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        self._check_date_rollover()
        
        if self.kill_switch_active:
            return False
        
        # Check daily loss
        if self.daily_stats.pnl < self.daily_loss_limit:
            self.trigger_kill_switch(f"Daily loss limit breached: {self.daily_stats.pnl*100:.2f}%")
            return False
        
        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded")
            return False
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if under rate limit."""
        now = datetime.now(timezone.utc).timestamp()
        
        # Remove old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts < 60
        ]
        
        if len(self.request_timestamps) >= self.rate_limit_per_minute:
            return False
        
        self.request_timestamps.append(now)
        return True
    
    def record_fill(
        self,
        expected_price: float,
        actual_price: float,
        quantity: float,
        side: str
    ):
        """Record a fill and update stats."""
        self._check_date_rollover()
        
        # Calculate slippage
        if side == "buy":
            slippage_bps = ((actual_price - expected_price) / expected_price) * 10000
        else:
            slippage_bps = ((expected_price - actual_price) / expected_price) * 10000
        
        self.daily_stats.orders_filled += 1
        self.daily_stats.slippage_total_bps += slippage_bps
        self.daily_stats.max_slippage_bps = max(self.daily_stats.max_slippage_bps, slippage_bps)
        
        # Check slippage limit
        if slippage_bps > self.slippage_limit_bps:
            self.trigger_kill_switch(f"Slippage breach: {slippage_bps:.1f} bps > {self.slippage_limit_bps} bps")
    
    def record_pnl(self, pnl: float):
        """Record daily P&L."""
        self._check_date_rollover()
        self.daily_stats.pnl = pnl
        
        if pnl < self.daily_loss_limit:
            self.trigger_kill_switch(f"Daily loss: {pnl*100:.2f}%")
    
    def check_order_size(
        self,
        symbol: str,
        quantity: float,
        adv: float,  # Average daily volume
        max_adv_pct: float = 0.01  # Max 1% of ADV
    ) -> tuple:
        """
        Check if order size is within ADV limits.
        Returns (allowed, reason).
        """
        if adv <= 0:
            return True, "ADV unknown - proceeding with caution"
        
        pct_of_adv = quantity / adv
        
        if pct_of_adv > max_adv_pct:
            return False, f"Order {pct_of_adv*100:.2f}% of ADV exceeds {max_adv_pct*100:.1f}% limit"
        
        return True, f"Order is {pct_of_adv*100:.2f}% of ADV"
    
    def trigger_kill_switch(self, reason: str):
        """Activate kill switch."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
    
    def reset_kill_switch(self):
        """Reset kill switch (requires confirmation in production)."""
        if os.environ.get("ALLOW_KILL_SWITCH_RESET") != "true":
            logger.warning("Kill switch reset blocked - set ALLOW_KILL_SWITCH_RESET=true")
            return False
        
        self.kill_switch_active = False
        self.kill_switch_reason = None
        logger.info("Kill switch reset")
        return True
    
    def get_block_reason(self) -> str:
        """Get reason trading is blocked."""
        if self.kill_switch_active:
            return f"Kill switch: {self.kill_switch_reason}"
        return "Unknown"
    
    def get_status(self) -> Dict:
        """Get current safety status."""
        self._check_date_rollover()
        
        return {
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "daily_pnl": self.daily_stats.pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "orders_placed": self.daily_stats.orders_placed,
            "orders_filled": self.daily_stats.orders_filled,
            "avg_slippage_bps": (
                self.daily_stats.slippage_total_bps / self.daily_stats.orders_filled
                if self.daily_stats.orders_filled > 0 else 0
            ),
            "max_slippage_bps": self.daily_stats.max_slippage_bps,
            "slippage_limit_bps": self.slippage_limit_bps
        }


if __name__ == "__main__":
    monitor = SafetyMonitor()
    
    # Test can_trade
    print(f"Can trade: {monitor.can_trade()}")
    
    # Test slippage
    monitor.record_fill(100.0, 100.05, 100, "buy")
    print(f"Status: {monitor.get_status()}")
    
    # Test kill switch
    monitor.trigger_kill_switch("Manual test")
    print(f"Can trade: {monitor.can_trade()}")
