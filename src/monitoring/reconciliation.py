"""
Daily Reality Reconciliation

Compares expected vs actual decisions:
- Regime classification alignment
- Slippage drift tracking
- Missed exit detection

Generates a daily markdown report.
"""

import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A single decision record."""
    timestamp: str
    symbol: str
    expected_action: str  # ENTER_LONG, EXIT_TO_CASH, HOLD
    actual_action: str
    expected_regime: str
    actual_regime: str
    expected_price: float
    actual_price: float
    slippage_bps: float
    
    @property
    def is_aligned(self) -> bool:
        return self.expected_action == self.actual_action
    
    @property
    def regime_aligned(self) -> bool:
        return self.expected_regime == self.actual_regime


@dataclass
class ReconciliationReport:
    """Daily reconciliation summary."""
    date: date
    total_decisions: int
    aligned_decisions: int
    regime_misclassifications: int
    avg_slippage_bps: float
    max_slippage_bps: float
    missed_exits: int
    missed_entries: int
    discrepancies: List[Dict]


class DailyReconciliation:
    """
    Reconciles expected vs actual trading behavior.
    
    Usage:
        recon = DailyReconciliation()
        recon.record_expected("SPY", "ENTER_LONG", "NEUTRAL", 450.0)
        recon.record_actual("SPY", "ENTER_LONG", "NEUTRAL", 450.05)
        report = recon.generate_report()
    """
    
    def __init__(self, reports_dir: Optional[Path] = None):
        project_root = Path(__file__).parent.parent.parent
        self.reports_dir = reports_dir or project_root / "reports" / "reconciliation"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.expected: Dict[str, Dict] = {}
        self.actual: Dict[str, Dict] = {}
        self.decisions: List[DecisionRecord] = []
        
        logger.info("DailyReconciliation initialized")
    
    def record_expected(
        self,
        symbol: str,
        action: str,
        regime: str,
        price: float,
        timestamp: Optional[str] = None
    ):
        """Record expected decision from backtest/model."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        key = f"{symbol}_{ts[:10]}"
        
        self.expected[key] = {
            "timestamp": ts,
            "symbol": symbol,
            "action": action,
            "regime": regime,
            "price": price
        }
    
    def record_actual(
        self,
        symbol: str,
        action: str,
        regime: str,
        price: float,
        timestamp: Optional[str] = None
    ):
        """Record actual decision from live execution."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        key = f"{symbol}_{ts[:10]}"
        
        self.actual[key] = {
            "timestamp": ts,
            "symbol": symbol,
            "action": action,
            "regime": regime,
            "price": price
        }
        
        # Match with expected
        if key in self.expected:
            exp = self.expected[key]
            
            # Calculate slippage
            if exp["price"] > 0:
                if action in ["ENTER_LONG"]:
                    slippage = ((price - exp["price"]) / exp["price"]) * 10000
                else:
                    slippage = ((exp["price"] - price) / exp["price"]) * 10000
            else:
                slippage = 0
            
            record = DecisionRecord(
                timestamp=ts,
                symbol=symbol,
                expected_action=exp["action"],
                actual_action=action,
                expected_regime=exp["regime"],
                actual_regime=regime,
                expected_price=exp["price"],
                actual_price=price,
                slippage_bps=slippage
            )
            self.decisions.append(record)
    
    def generate_report(self) -> ReconciliationReport:
        """Generate daily reconciliation report."""
        today = date.today()
        
        if not self.decisions:
            return ReconciliationReport(
                date=today,
                total_decisions=0,
                aligned_decisions=0,
                regime_misclassifications=0,
                avg_slippage_bps=0,
                max_slippage_bps=0,
                missed_exits=0,
                missed_entries=0,
                discrepancies=[]
            )
        
        aligned = sum(1 for d in self.decisions if d.is_aligned)
        regime_misclass = sum(1 for d in self.decisions if not d.regime_aligned)
        slippages = [d.slippage_bps for d in self.decisions]
        
        # Missed exits/entries
        missed_exits = sum(1 for d in self.decisions 
                          if d.expected_action == "EXIT_TO_CASH" 
                          and d.actual_action != "EXIT_TO_CASH")
        missed_entries = sum(1 for d in self.decisions 
                            if d.expected_action == "ENTER_LONG" 
                            and d.actual_action != "ENTER_LONG")
        
        # Discrepancies
        discrepancies = []
        for d in self.decisions:
            if not d.is_aligned or not d.regime_aligned or abs(d.slippage_bps) > 20:
                discrepancies.append({
                    "timestamp": d.timestamp,
                    "symbol": d.symbol,
                    "issue": self._describe_issue(d),
                    "expected": d.expected_action,
                    "actual": d.actual_action,
                    "slippage": d.slippage_bps
                })
        
        return ReconciliationReport(
            date=today,
            total_decisions=len(self.decisions),
            aligned_decisions=aligned,
            regime_misclassifications=regime_misclass,
            avg_slippage_bps=sum(slippages) / len(slippages),
            max_slippage_bps=max(slippages),
            missed_exits=missed_exits,
            missed_entries=missed_entries,
            discrepancies=discrepancies
        )
    
    def _describe_issue(self, d: DecisionRecord) -> str:
        """Describe the discrepancy."""
        issues = []
        if not d.is_aligned:
            issues.append(f"Action mismatch: {d.expected_action} → {d.actual_action}")
        if not d.regime_aligned:
            issues.append(f"Regime mismatch: {d.expected_regime} → {d.actual_regime}")
        if abs(d.slippage_bps) > 20:
            issues.append(f"High slippage: {d.slippage_bps:.1f} bps")
        return "; ".join(issues) if issues else "Minor"
    
    def save_report(self, report: ReconciliationReport) -> Path:
        """Save report to markdown file."""
        filename = self.reports_dir / f"recon_{report.date.isoformat()}.md"
        
        content = f"""# Daily Reality Reconciliation Report

**Date**: {report.date}
**Generated**: {datetime.now(timezone.utc).isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total Decisions | {report.total_decisions} |
| Aligned | {report.aligned_decisions} ({report.aligned_decisions/max(1,report.total_decisions)*100:.0f}%) |
| Regime Misclassifications | {report.regime_misclassifications} |
| Avg Slippage | {report.avg_slippage_bps:.1f} bps |
| Max Slippage | {report.max_slippage_bps:.1f} bps |
| Missed Exits | {report.missed_exits} |
| Missed Entries | {report.missed_entries} |

## Discrepancies

"""
        if report.discrepancies:
            content += "| Time | Symbol | Issue | Expected | Actual | Slippage |\n"
            content += "|------|--------|-------|----------|--------|----------|\n"
            for d in report.discrepancies:
                content += f"| {d['timestamp'][:19]} | {d['symbol']} | {d['issue']} | {d['expected']} | {d['actual']} | {d['slippage']:.1f} bps |\n"
        else:
            content += "_No discrepancies detected._\n"
        
        content += """
## Status

"""
        if report.regime_misclassifications > 0:
            content += "> [!WARNING]\n> Regime misclassifications detected. Review model inputs.\n"
        if report.missed_exits > 0:
            content += "> [!CAUTION]\n> Missed exits detected. Check kill-switch triggers.\n"
        if report.aligned_decisions == report.total_decisions:
            content += "> [!NOTE]\n> Perfect alignment achieved.\n"
        
        with open(filename, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved: {filename}")
        return filename


if __name__ == "__main__":
    recon = DailyReconciliation()
    
    # Simulate some decisions
    recon.record_expected("SPY", "ENTER_LONG", "NEUTRAL", 450.0)
    recon.record_actual("SPY", "ENTER_LONG", "NEUTRAL", 450.10)
    
    recon.record_expected("GLD", "EXIT_TO_CASH", "HOSTILE", 180.0)
    recon.record_actual("GLD", "HOLD", "NEUTRAL", 179.5)  # Mismatch!
    
    report = recon.generate_report()
    print(f"Aligned: {report.aligned_decisions}/{report.total_decisions}")
    print(f"Regime misclass: {report.regime_misclassifications}")
    
    recon.save_report(report)
