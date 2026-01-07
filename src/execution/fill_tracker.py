"""
Fill Tracker

Tracks order fills, calculates slippage, and builds live slippage distribution.
Used for reality reconciliation between backtest assumptions and live execution.
"""

import os
import sys
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """Complete fill record for a trade."""
    timestamp: str
    order_id: str
    symbol: str
    side: str  # buy/sell
    expected_price: float
    actual_fill_price: float
    slippage_bps: float
    qty: float
    notional: float
    time_delay_ms: Optional[float]  # submission to fill
    partial_fill: bool


class FillTracker:
    """
    Tracks fills and builds slippage distribution.
    
    Usage:
        tracker = FillTracker()
        tracker.record_fill(...)
        stats = tracker.get_slippage_stats()
        tracker.save_report()
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize fill tracker.
        
        Args:
            log_dir: Directory for fill logs
        """
        self.log_dir = log_dir or Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.fills_path = self.log_dir / "fills.csv"
        self.fills: List[FillRecord] = []
        
        # Load existing fills
        self._load_fills()
        
        logger.info(f"FillTracker initialized. {len(self.fills)} historical fills loaded.")
    
    def _load_fills(self):
        """Load existing fills from disk."""
        if self.fills_path.exists():
            df = pd.read_csv(self.fills_path)
            for _, row in df.iterrows():
                self.fills.append(FillRecord(
                    timestamp=row['timestamp'],
                    order_id=row['order_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    expected_price=row['expected_price'],
                    actual_fill_price=row['actual_fill_price'],
                    slippage_bps=row['slippage_bps'],
                    qty=row['qty'],
                    notional=row['notional'],
                    time_delay_ms=row.get('time_delay_ms'),
                    partial_fill=row.get('partial_fill', False)
                ))
    
    def _save_fills(self):
        """Save all fills to disk."""
        if not self.fills:
            return
        
        df = pd.DataFrame([asdict(f) for f in self.fills])
        df.to_csv(self.fills_path, index=False)
    
    def record_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        actual_fill_price: float,
        qty: float,
        submit_time: Optional[datetime] = None,
        fill_time: Optional[datetime] = None,
        partial_fill: bool = False
    ) -> FillRecord:
        """
        Record a new fill.
        
        Args:
            order_id: Broker order ID
            symbol: Trading symbol
            side: "buy" or "sell"
            expected_price: Price assumption from model
            actual_fill_price: Actual execution price
            qty: Filled quantity
            submit_time: Order submission time
            fill_time: Order fill time
            partial_fill: Whether this was a partial fill
            
        Returns:
            FillRecord created
        """
        # Calculate slippage
        if expected_price > 0:
            slippage_bps = ((actual_fill_price - expected_price) / expected_price) * 10000
            # Adjust sign for sells (negative slippage = good for sells)
            if side == "sell":
                slippage_bps = -slippage_bps
        else:
            slippage_bps = 0.0
        
        # Calculate time delay
        time_delay_ms = None
        if submit_time and fill_time:
            delta = (fill_time - submit_time).total_seconds() * 1000
            time_delay_ms = delta
        
        record = FillRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            order_id=order_id,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_fill_price=actual_fill_price,
            slippage_bps=slippage_bps,
            qty=qty,
            notional=actual_fill_price * qty,
            time_delay_ms=time_delay_ms,
            partial_fill=partial_fill
        )
        
        self.fills.append(record)
        self._save_fills()
        
        logger.info(f"Fill recorded: {symbol} {side} @ {actual_fill_price:.2f} (slip: {slippage_bps:.1f}bps)")
        
        return record
    
    def get_slippage_stats(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get slippage statistics.
        
        Args:
            symbol: Filter by symbol (None for all)
            
        Returns:
            Dictionary with mean, std, min, max, count
        """
        fills = self.fills
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        
        if not fills:
            return {
                "mean_bps": 0.0,
                "std_bps": 0.0,
                "min_bps": 0.0,
                "max_bps": 0.0,
                "count": 0,
                "total_notional": 0.0
            }
        
        slippages = [f.slippage_bps for f in fills]
        
        return {
            "mean_bps": float(np.mean(slippages)),
            "std_bps": float(np.std(slippages)),
            "min_bps": float(np.min(slippages)),
            "max_bps": float(np.max(slippages)),
            "count": len(fills),
            "total_notional": sum(f.notional for f in fills),
            "partial_fill_rate": sum(1 for f in fills if f.partial_fill) / len(fills)
        }
    
    def check_slippage_bounds(
        self,
        expected_mean: float = 5.0,
        expected_std: float = 5.0,
        tolerance_std: float = 1.0
    ) -> bool:
        """
        Check if slippage is within expected bounds.
        
        Args:
            expected_mean: Expected slippage mean (bps)
            expected_std: Expected slippage std (bps)
            tolerance_std: Number of std deviations allowed
            
        Returns:
            True if within bounds, False otherwise
        """
        stats = self.get_slippage_stats()
        
        if stats["count"] < 5:
            logger.warning("Not enough fills to evaluate slippage bounds")
            return True  # Not enough data
        
        # Check if mean is within tolerance
        upper_bound = expected_mean + (tolerance_std * expected_std)
        lower_bound = expected_mean - (tolerance_std * expected_std)
        
        within_bounds = lower_bound <= stats["mean_bps"] <= upper_bound
        
        if not within_bounds:
            logger.warning(
                f"Slippage out of bounds! Mean: {stats['mean_bps']:.1f}bps, "
                f"Expected: {expected_mean:.1f}±{expected_std * tolerance_std:.1f}bps"
            )
        
        return within_bounds
    
    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Save slippage report to JSON.
        
        Returns:
            Path to saved report
        """
        output_path = output_path or self.log_dir.parent / "reports" / "slippage_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_slippage_stats()
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": stats,
            "fills": [asdict(f) for f in self.fills[-100:]]  # Last 100 fills
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Slippage report saved: {output_path}")
        return output_path
    
    def get_slippage_distribution(self) -> pd.Series:
        """Get full slippage distribution as pandas Series."""
        if not self.fills:
            return pd.Series(dtype=float)
        return pd.Series([f.slippage_bps for f in self.fills])


if __name__ == "__main__":
    # Quick test
    tracker = FillTracker()
    
    # Simulate some fills
    tracker.record_fill(
        order_id="test_001",
        symbol="SPY",
        side="buy",
        expected_price=500.00,
        actual_fill_price=500.05,
        qty=100
    )
    
    tracker.record_fill(
        order_id="test_002",
        symbol="SPY",
        side="sell",
        expected_price=501.00,
        actual_fill_price=500.95,
        qty=100
    )
    
    stats = tracker.get_slippage_stats()
    print(f"Slippage Stats: {stats}")
    
    report_path = tracker.save_report()
    print(f"Report saved: {report_path}")
