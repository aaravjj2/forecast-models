#!/usr/bin/env python3
"""
Paper Trading Runner

Main entry point for paper trading the Volatility-Gated Long Exposure strategy.

Usage:
    # Run once (for cron/scheduler)
    python run_paper_trading.py --once
    
    # Run continuously (checks every 5 minutes during market hours)
    python run_paper_trading.py --continuous
    
    # Train models first
    python run_paper_trading.py --train

Environment Variables Required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from execution.alpaca_adapter import AlpacaAdapter, ExposureState
from execution.order_manager import OrderManager
from execution.live_signal_generator import LiveSignalGenerator
from execution.fill_tracker import FillTracker
from monitoring.kill_switches import KillSwitches, KillSwitchConfig
from monitoring.audit_logger import AuditLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """
    Orchestrates paper trading execution.
    
    Flow:
    1. Check kill switches
    2. Generate signal from regime models
    3. Execute signal via order manager
    4. Track fills and slippage
    5. Log everything
    """
    
    def __init__(self, symbol: str = "SPY"):
        """
        Initialize paper trading runner.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        
        # Initialize components
        logger.info("Initializing Paper Trading Runner...")
        
        self.adapter = AlpacaAdapter()
        self.order_manager = OrderManager(symbol=symbol, adapter=self.adapter)
        self.signal_generator = LiveSignalGenerator(symbol=symbol)
        self.fill_tracker = FillTracker()
        self.kill_switches = KillSwitches(
            config=KillSwitchConfig(
                max_daily_loss_pct=3.0,
                min_regime_confidence=0.3,
                max_consecutive_api_errors=3,
                max_slippage_multiple=3.0,
                expected_slippage_bps=5.0
            )
        )
        self.audit_logger = AuditLogger()
        
        # State tracking
        self.expected_position: float = 0.0
        self.last_signal_time: datetime = None
        
        logger.info(f"Paper Trading Runner initialized for {symbol}")
    
    def check_market_hours(self) -> bool:
        """Check if market is open."""
        return self.adapter.is_market_open()
    
    def run_once(self):
        """
        Run one iteration of the trading loop.
        
        This is the main trading logic.
        """
        now = datetime.now(timezone.utc)
        logger.info(f"=== Trading Loop Start: {now.isoformat()} ===")
        
        # 1. Get current account state
        try:
            state = self.adapter.get_account_state(self.symbol)
            api_success = True
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            api_success = False
            state = None
        
        if state is None:
            # Can't proceed without state
            self.kill_switches.check_all(
                current_equity=0,
                regime_confidence=0,
                current_slippage_bps=0,
                current_position=0,
                expected_position=0,
                api_success=False
            )
            return
        
        logger.info(f"Account: ${state.equity:.2f} equity, {state.exposure.value}")
        
        # 2. Generate signal
        current_exposure = "LONG" if state.exposure == ExposureState.LONG else "FLAT"
        signal_result = self.signal_generator.generate_signal(current_exposure=current_exposure)
        
        logger.info(f"Signal: {signal_result.signal} (State: {signal_result.lattice_state})")
        logger.info(f"Regime Probs: {signal_result.regime_probs}")
        
        # 3. Check kill switches
        slippage_stats = self.fill_tracker.get_slippage_stats()
        last_slippage = slippage_stats.get('mean_bps', 0)
        
        ks_result = self.kill_switches.check_all(
            current_equity=state.equity,
            regime_confidence=signal_result.confidence,
            current_slippage_bps=last_slippage,
            current_position=state.position_qty,
            expected_position=self.expected_position,
            api_success=api_success
        )
        
        # 4. Determine action
        if ks_result.triggered:
            logger.warning(f"Kill switch triggered: {ks_result.reason}")
            
            if ks_result.action in ["flatten", "halt"]:
                # Emergency flatten
                trade_record = self.order_manager.emergency_flatten(
                    reason=f"kill_switch:{ks_result.switch_type.value}"
                )
                self.expected_position = 0.0
                
                # Log to audit
                self.audit_logger.log_decision(
                    vol_prob=signal_result.regime_probs.get('vol_high_prob', 0),
                    trend_prob=signal_result.regime_probs.get('trend_robust_prob', 0),
                    liq_prob=signal_result.regime_probs.get('liq_stressed_prob', 0),
                    lattice_state=str(signal_result.lattice_state),
                    regime_confidence=signal_result.confidence,
                    signal="EXIT_TO_CASH",
                    signal_source=f"kill_switch:{ks_result.switch_type.value}",
                    order_id=trade_record.order_id if trade_record else None,
                    order_status=trade_record.order_status if trade_record else None,
                    notes=ks_result.reason
                )
            
            return
        
        # 5. Execute signal
        trade_record = self.order_manager.execute_signal(
            signal=signal_result.signal,
            signal_source="regime_model",
            notes=signal_result.notes
        )
        
        # Update expected position
        if signal_result.signal == "ENTER_LONG":
            # Estimate position from buying power
            price = self.adapter.get_latest_price(self.symbol)
            self.expected_position = (state.buying_power * 0.95) / price
        elif signal_result.signal == "EXIT_TO_CASH":
            self.expected_position = 0.0
        
        # 6. Log to audit
        self.audit_logger.log_decision(
            vol_prob=signal_result.regime_probs.get('vol_high_prob', 0),
            trend_prob=signal_result.regime_probs.get('trend_robust_prob', 0),
            liq_prob=signal_result.regime_probs.get('liq_stressed_prob', 0),
            lattice_state=str(signal_result.lattice_state),
            regime_confidence=signal_result.confidence,
            signal=signal_result.signal,
            signal_source="regime_model",
            order_id=trade_record.order_id if trade_record else None,
            order_status=trade_record.order_status if trade_record else None,
            slippage_bps=trade_record.slippage_bps if trade_record else None,
            notes=signal_result.notes
        )
        
        self.last_signal_time = now
        logger.info(f"=== Trading Loop Complete ===\n")
    
    def run_continuous(self, check_interval_minutes: int = 5):
        """
        Run continuously during market hours.
        
        Args:
            check_interval_minutes: Interval between checks
        """
        logger.info("Starting continuous paper trading...")
        
        while True:
            try:
                if self.check_market_hours():
                    self.run_once()
                else:
                    logger.info("Market closed. Waiting...")
                
                # Sleep
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Interrupted. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def train_models(self):
        """Train regime models before starting trading."""
        logger.info("Training regime models...")
        self.signal_generator.train_models()
        logger.info("Model training complete.")
    
    def generate_daily_report(self) -> Path:
        """Generate end-of-day report."""
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Slippage report
        slippage_path = self.fill_tracker.save_report(
            reports_dir / f"slippage_{date_str}.json"
        )
        
        # Audit report
        audit_path = self.audit_logger.export_report(
            reports_dir / f"audit_{date_str}.json"
        )
        
        # Combined summary
        state = self.adapter.get_account_state(self.symbol)
        slippage_stats = self.fill_tracker.get_slippage_stats()
        
        summary = {
            "date": date_str,
            "equity": state.equity,
            "exposure": state.exposure.value,
            "unrealized_pnl": state.unrealized_pnl,
            "slippage_mean_bps": slippage_stats.get('mean_bps', 0),
            "slippage_std_bps": slippage_stats.get('std_bps', 0),
            "total_trades": slippage_stats.get('count', 0)
        }
        
        summary_path = reports_dir / f"daily_summary_{date_str}.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Daily report generated: {summary_path}")
        return summary_path


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument("--symbol", default="SPY", help="Trading symbol")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--train", action="store_true", help="Train models first")
    parser.add_argument("--report", action="store_true", help="Generate daily report")
    
    args = parser.parse_args()
    
    try:
        runner = PaperTradingRunner(symbol=args.symbol)
        
        if args.train:
            runner.train_models()
        
        if args.report:
            runner.generate_daily_report()
            return
        
        if args.continuous:
            runner.run_continuous()
        elif args.once:
            runner.run_once()
        else:
            # Default: run once
            runner.run_once()
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
        sys.exit(1)


if __name__ == "__main__":
    main()
