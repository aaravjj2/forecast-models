"""
Shadow Backtesting

Runs backtest in parallel with live data:
- Asserts action parity (backtest == live)
- Logs divergences with timestamps
- No alpha generation, only validation

Paper-first. Confidence building only.
"""

import logging
from datetime import datetime, timezone, date
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    """Result of shadow comparison."""
    timestamp: str
    symbol: str
    live_action: str
    shadow_action: str
    is_aligned: bool
    divergence_reason: Optional[str] = None


class ShadowBacktester:
    """
    Runs backtest logic in parallel with live execution.
    
    Detects divergences between what the backtest would do vs
    what the live system actually does.
    
    Usage:
        shadow = ShadowBacktester(regime_model, strategy)
        
        # During live execution:
        shadow.record_live("SPY", "ENTER_LONG", price_data)
        shadow.record_shadow("SPY", price_data)  # Runs backtest logic
        
        divergences = shadow.get_divergences()
    """
    
    def __init__(
        self,
        regime_predictor: Optional[Callable] = None,
        action_decider: Optional[Callable] = None,
        log_dir: Optional[Path] = None
    ):
        project_root = Path(__file__).parent.parent.parent
        self.log_dir = log_dir or project_root / "logs" / "shadow"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.regime_predictor = regime_predictor
        self.action_decider = action_decider
        
        self.live_actions: Dict[str, str] = {}
        self.shadow_actions: Dict[str, str] = {}
        self.results: List[ShadowResult] = []
        
        logger.info("ShadowBacktester initialized")
    
    def record_live(
        self,
        symbol: str,
        action: str,
        timestamp: Optional[str] = None
    ):
        """Record what the live system actually did."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        key = f"{symbol}_{ts[:10]}"
        
        self.live_actions[key] = {
            "action": action,
            "timestamp": ts
        }
        
        # Check for shadow match
        if key in self.shadow_actions:
            self._compare(key, ts, symbol)
    
    def record_shadow(
        self,
        symbol: str,
        features: Dict,
        timestamp: Optional[str] = None
    ):
        """
        Run backtest logic and record what it would do.
        
        Args:
            symbol: Trading symbol
            features: Feature dict for regime prediction
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        key = f"{symbol}_{ts[:10]}"
        
        # Run shadow regime prediction
        if self.regime_predictor and self.action_decider:
            regime = self.regime_predictor(features)
            action = self.action_decider(regime)
        else:
            # Mock for testing
            regime = "NEUTRAL"
            action = "HOLD"
        
        self.shadow_actions[key] = {
            "action": action,
            "regime": regime,
            "timestamp": ts
        }
        
        # Check for live match
        if key in self.live_actions:
            self._compare(key, ts, symbol)
    
    def _compare(self, key: str, timestamp: str, symbol: str):
        """Compare live vs shadow actions."""
        live = self.live_actions.get(key, {})
        shadow = self.shadow_actions.get(key, {})
        
        live_action = live.get("action", "UNKNOWN")
        shadow_action = shadow.get("action", "UNKNOWN")
        
        is_aligned = live_action == shadow_action
        
        divergence_reason = None
        if not is_aligned:
            divergence_reason = f"Live={live_action}, Shadow={shadow_action}"
            logger.warning(f"DIVERGENCE [{symbol}]: {divergence_reason}")
        
        result = ShadowResult(
            timestamp=timestamp,
            symbol=symbol,
            live_action=live_action,
            shadow_action=shadow_action,
            is_aligned=is_aligned,
            divergence_reason=divergence_reason
        )
        self.results.append(result)
    
    def get_divergences(self) -> List[ShadowResult]:
        """Get all divergences."""
        return [r for r in self.results if not r.is_aligned]
    
    def get_alignment_rate(self) -> float:
        """Get percentage of aligned actions."""
        if not self.results:
            return 1.0
        aligned = sum(1 for r in self.results if r.is_aligned)
        return aligned / len(self.results)
    
    def assert_parity(self) -> bool:
        """Assert 100% parity between live and shadow."""
        divergences = self.get_divergences()
        if divergences:
            for d in divergences:
                logger.error(f"Parity assertion failed: {d.divergence_reason}")
            return False
        return True
    
    def save_divergence_log(self) -> Path:
        """Save divergences to log file."""
        today = date.today()
        filename = self.log_dir / f"shadow_divergences_{today.isoformat()}.log"
        
        with open(filename, 'w') as f:
            f.write(f"Shadow Backtest Divergence Log\n")
            f.write(f"Date: {today}\n")
            f.write(f"Total comparisons: {len(self.results)}\n")
            f.write(f"Alignment rate: {self.get_alignment_rate()*100:.1f}%\n")
            f.write("\n--- Divergences ---\n\n")
            
            for d in self.get_divergences():
                f.write(f"[{d.timestamp}] {d.symbol}: {d.divergence_reason}\n")
        
        logger.info(f"Divergence log saved: {filename}")
        return filename


class ShadowRunner:
    """
    Continuous shadow runner for parallel execution.
    """
    
    def __init__(self, shadow_backtester: ShadowBacktester):
        self.shadow = shadow_backtester
        self.running = False
    
    def process_signal(
        self,
        symbol: str,
        live_action: str,
        features: Dict
    ) -> ShadowResult:
        """
        Process a signal through both live and shadow paths.
        
        Returns the comparison result.
        """
        ts = datetime.now(timezone.utc).isoformat()
        
        # Record live
        self.shadow.record_live(symbol, live_action, ts)
        
        # Record shadow
        self.shadow.record_shadow(symbol, features, ts)
        
        # Get latest result
        if self.shadow.results:
            return self.shadow.results[-1]
        return None


if __name__ == "__main__":
    shadow = ShadowBacktester()
    
    # Simulate aligned actions
    shadow.record_live("SPY", "ENTER_LONG")
    shadow.record_shadow("SPY", {})
    shadow.shadow_actions[list(shadow.shadow_actions.keys())[-1]]["action"] = "ENTER_LONG"
    shadow._compare(list(shadow.live_actions.keys())[-1], 
                   datetime.now(timezone.utc).isoformat(), "SPY")
    
    # Simulate divergent action
    shadow.record_live("GLD", "EXIT_TO_CASH")
    shadow.record_shadow("GLD", {})
    # Shadow defaults to HOLD, creating divergence
    
    print(f"Alignment rate: {shadow.get_alignment_rate()*100:.1f}%")
    print(f"Divergences: {len(shadow.get_divergences())}")
    
    shadow.save_divergence_log()
