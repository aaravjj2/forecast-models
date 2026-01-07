"""
Kill Switches

MANDATORY safety mechanisms that override ALL trading logic.
These protect capital in adverse conditions.

Kill Switch Conditions:
1. Daily loss > X% → Exit to cash
2. Regime model confidence collapses → Force flat
3. Broker API error (consecutive) → Flat + halt
4. Slippage > 3× expected → Halt trading
5. Unexpected position detected → Emergency close
"""

import os
import sys
import logging
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KillSwitchType(Enum):
    """Types of kill switches."""
    DAILY_LOSS = "daily_loss"
    CONFIDENCE_COLLAPSE = "confidence_collapse"
    API_ERROR = "api_error"
    SLIPPAGE_BREACH = "slippage_breach"
    UNEXPECTED_POSITION = "unexpected_position"


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch thresholds."""
    max_daily_loss_pct: float = 3.0  # Max % loss in a day
    min_regime_confidence: float = 0.3  # Min confidence before collapse
    max_consecutive_api_errors: int = 3
    max_slippage_multiple: float = 3.0  # Max slippage as multiple of expected
    expected_slippage_bps: float = 5.0  # Expected slippage in bps


@dataclass
class KillSwitchState:
    """Current state of kill switch monitoring."""
    is_halted: bool = False
    halt_reason: Optional[str] = None
    halt_time: Optional[datetime] = None
    consecutive_api_errors: int = 0
    daily_pnl_pct: float = 0.0
    last_check: Optional[datetime] = None


@dataclass
class KillSwitchResult:
    """Result of a kill switch check."""
    triggered: bool
    switch_type: Optional[KillSwitchType]
    action: str  # "none", "flatten", "halt"
    reason: str
    timestamp: datetime


class KillSwitches:
    """
    Kill switch manager.
    
    IMPORTANT: Kill switches OVERRIDE all trading logic.
    
    Usage:
        ks = KillSwitches()
        result = ks.check_all(...)
        if result.triggered:
            order_manager.emergency_flatten(result.reason)
    """
    
    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        on_trigger: Optional[Callable[[KillSwitchResult], None]] = None
    ):
        """
        Initialize kill switches.
        
        Args:
            config: Kill switch configuration
            on_trigger: Callback when a kill switch is triggered
        """
        self.config = config or KillSwitchConfig()
        self.state = KillSwitchState()
        self.on_trigger = on_trigger
        
        # Daily tracking
        self._daily_start_equity: Optional[float] = None
        self._daily_date: Optional[date] = None
        
        logger.info("KillSwitches initialized")
        logger.info(f"  Max daily loss: {self.config.max_daily_loss_pct}%")
        logger.info(f"  Min confidence: {self.config.min_regime_confidence}")
        logger.info(f"  Max API errors: {self.config.max_consecutive_api_errors}")
        logger.info(f"  Max slippage: {self.config.max_slippage_multiple}x expected")
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking. Call at market open."""
        self._daily_start_equity = current_equity
        self._daily_date = date.today()
        self.state.daily_pnl_pct = 0.0
        logger.info(f"Daily tracking reset. Start equity: ${current_equity:.2f}")
    
    def check_all(
        self,
        current_equity: float,
        regime_confidence: float,
        current_slippage_bps: float,
        current_position: float,  # shares
        expected_position: float,  # shares
        api_success: bool = True
    ) -> KillSwitchResult:
        """
        Check ALL kill switches.
        
        Args:
            current_equity: Current account equity
            regime_confidence: Current regime model confidence (0-1)
            current_slippage_bps: Most recent slippage in bps
            current_position: Actual position (shares)
            expected_position: Expected position (shares)
            api_success: Whether last API call succeeded
            
        Returns:
            KillSwitchResult indicating if any switch triggered
        """
        self.state.last_check = datetime.now(timezone.utc)
        
        # If already halted, stay halted
        if self.state.is_halted:
            return KillSwitchResult(
                triggered=True,
                switch_type=None,
                action="halt",
                reason=f"System halted since {self.state.halt_time}: {self.state.halt_reason}",
                timestamp=self.state.last_check
            )
        
        # Check each kill switch
        result = self._check_daily_loss(current_equity)
        if result.triggered:
            return self._handle_trigger(result)
        
        result = self._check_confidence(regime_confidence)
        if result.triggered:
            return self._handle_trigger(result)
        
        result = self._check_api_errors(api_success)
        if result.triggered:
            return self._handle_trigger(result)
        
        result = self._check_slippage(current_slippage_bps)
        if result.triggered:
            return self._handle_trigger(result)
        
        result = self._check_position(current_position, expected_position)
        if result.triggered:
            return self._handle_trigger(result)
        
        # All clear
        return KillSwitchResult(
            triggered=False,
            switch_type=None,
            action="none",
            reason="All checks passed",
            timestamp=self.state.last_check
        )
    
    def _check_daily_loss(self, current_equity: float) -> KillSwitchResult:
        """Check if daily loss exceeds threshold."""
        now = datetime.now(timezone.utc)
        
        # Reset if new day
        if self._daily_date != date.today():
            self.reset_daily(current_equity)
        
        if self._daily_start_equity is None or self._daily_start_equity == 0:
            return KillSwitchResult(
                triggered=False,
                switch_type=KillSwitchType.DAILY_LOSS,
                action="none",
                reason="No starting equity recorded",
                timestamp=now
            )
        
        pnl_pct = ((current_equity - self._daily_start_equity) / self._daily_start_equity) * 100
        self.state.daily_pnl_pct = pnl_pct
        
        if pnl_pct < -self.config.max_daily_loss_pct:
            return KillSwitchResult(
                triggered=True,
                switch_type=KillSwitchType.DAILY_LOSS,
                action="flatten",
                reason=f"Daily loss {pnl_pct:.2f}% exceeds -{self.config.max_daily_loss_pct}% limit",
                timestamp=now
            )
        
        return KillSwitchResult(
            triggered=False,
            switch_type=KillSwitchType.DAILY_LOSS,
            action="none",
            reason=f"Daily PnL: {pnl_pct:.2f}%",
            timestamp=now
        )
    
    def _check_confidence(self, confidence: float) -> KillSwitchResult:
        """Check if regime model confidence has collapsed."""
        now = datetime.now(timezone.utc)
        
        if confidence < self.config.min_regime_confidence:
            return KillSwitchResult(
                triggered=True,
                switch_type=KillSwitchType.CONFIDENCE_COLLAPSE,
                action="flatten",
                reason=f"Regime confidence {confidence:.2f} below {self.config.min_regime_confidence}",
                timestamp=now
            )
        
        return KillSwitchResult(
            triggered=False,
            switch_type=KillSwitchType.CONFIDENCE_COLLAPSE,
            action="none",
            reason=f"Confidence: {confidence:.2f}",
            timestamp=now
        )
    
    def _check_api_errors(self, api_success: bool) -> KillSwitchResult:
        """Check for consecutive API errors."""
        now = datetime.now(timezone.utc)
        
        if api_success:
            self.state.consecutive_api_errors = 0
        else:
            self.state.consecutive_api_errors += 1
        
        if self.state.consecutive_api_errors >= self.config.max_consecutive_api_errors:
            return KillSwitchResult(
                triggered=True,
                switch_type=KillSwitchType.API_ERROR,
                action="halt",
                reason=f"{self.state.consecutive_api_errors} consecutive API errors",
                timestamp=now
            )
        
        return KillSwitchResult(
            triggered=False,
            switch_type=KillSwitchType.API_ERROR,
            action="none",
            reason=f"API errors: {self.state.consecutive_api_errors}",
            timestamp=now
        )
    
    def _check_slippage(self, slippage_bps: float) -> KillSwitchResult:
        """Check if slippage exceeds threshold."""
        now = datetime.now(timezone.utc)
        
        max_allowed = self.config.expected_slippage_bps * self.config.max_slippage_multiple
        
        if abs(slippage_bps) > max_allowed:
            return KillSwitchResult(
                triggered=True,
                switch_type=KillSwitchType.SLIPPAGE_BREACH,
                action="halt",
                reason=f"Slippage {slippage_bps:.1f}bps exceeds {max_allowed:.1f}bps limit",
                timestamp=now
            )
        
        return KillSwitchResult(
            triggered=False,
            switch_type=KillSwitchType.SLIPPAGE_BREACH,
            action="none",
            reason=f"Slippage: {slippage_bps:.1f}bps",
            timestamp=now
        )
    
    def _check_position(self, actual: float, expected: float) -> KillSwitchResult:
        """Check for unexpected position."""
        now = datetime.now(timezone.utc)
        
        # Allow small tolerance for rounding
        tolerance = max(1, abs(expected) * 0.05)  # 5% or 1 share
        
        if abs(actual - expected) > tolerance:
            return KillSwitchResult(
                triggered=True,
                switch_type=KillSwitchType.UNEXPECTED_POSITION,
                action="flatten",
                reason=f"Position mismatch: actual={actual}, expected={expected}",
                timestamp=now
            )
        
        return KillSwitchResult(
            triggered=False,
            switch_type=KillSwitchType.UNEXPECTED_POSITION,
            action="none",
            reason=f"Position OK: {actual}",
            timestamp=now
        )
    
    def _handle_trigger(self, result: KillSwitchResult) -> KillSwitchResult:
        """Handle a triggered kill switch."""
        logger.error(f"KILL SWITCH TRIGGERED: {result.switch_type.value}")
        logger.error(f"  Reason: {result.reason}")
        logger.error(f"  Action: {result.action}")
        
        if result.action == "halt":
            self.state.is_halted = True
            self.state.halt_reason = result.reason
            self.state.halt_time = result.timestamp
        
        if self.on_trigger:
            self.on_trigger(result)
        
        return result
    
    def reset_halt(self, reason: str = "manual"):
        """Reset halt state (manual intervention only)."""
        logger.warning(f"Halt state reset by: {reason}")
        self.state.is_halted = False
        self.state.halt_reason = None
        self.state.halt_time = None
        self.state.consecutive_api_errors = 0


if __name__ == "__main__":
    # Test kill switches
    ks = KillSwitches()
    
    # Simulate normal conditions
    result = ks.check_all(
        current_equity=100000,
        regime_confidence=0.6,
        current_slippage_bps=3.0,
        current_position=100,
        expected_position=100,
        api_success=True
    )
    print(f"Normal: triggered={result.triggered}, reason={result.reason}")
    
    # Simulate daily loss
    ks.reset_daily(100000)
    result = ks.check_all(
        current_equity=96000,  # 4% loss
        regime_confidence=0.6,
        current_slippage_bps=3.0,
        current_position=100,
        expected_position=100,
        api_success=True
    )
    print(f"Daily Loss: triggered={result.triggered}, action={result.action}")
    
    # Reset and test slippage
    ks = KillSwitches()
    ks.reset_daily(100000)
    result = ks.check_all(
        current_equity=100000,
        regime_confidence=0.6,
        current_slippage_bps=20.0,  # 4x expected
        current_position=100,
        expected_position=100,
        api_success=True
    )
    print(f"Slippage: triggered={result.triggered}, action={result.action}")
