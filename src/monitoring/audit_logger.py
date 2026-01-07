"""
Audit Logger

Logs every trade decision for auditability.
This is NON-OPTIONAL - every decision must be recorded.

Logged Information:
- Timestamp (UTC)
- Regime probabilities
- Model version hash  
- Execution parameters
- Result
"""

import os
import sys
import json
import hashlib
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Complete audit record for a decision."""
    # Timing
    timestamp: str
    decision_id: str
    
    # Regime State
    vol_prob: float
    trend_prob: float
    liq_prob: float
    lattice_state: str
    regime_confidence: float
    
    # Decision
    signal: str  # ENTER_LONG, EXIT_TO_CASH, HOLD
    signal_source: str  # regime_model, kill_switch, manual
    
    # Execution
    order_type: Optional[str]
    order_qty: Optional[float]
    order_notional: Optional[float]
    
    # Result
    order_id: Optional[str]
    order_status: Optional[str]
    fill_price: Optional[float]
    slippage_bps: Optional[float]
    
    # Metadata
    model_version: str
    config_hash: str
    notes: str = ""


class AuditLogger:
    """
    Audit logger for complete decision trail.
    
    Usage:
        audit = AuditLogger()
        record = audit.log_decision(...)
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = log_dir or Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_log_path = self.log_dir / "trade_audit.csv"
        self.audit_json_path = self.log_dir / "trade_audit.json"
        
        # Get model version (git SHA)
        self.model_version = self._get_git_sha()
        
        # Initialize log files
        self._init_csv()
        
        # Decision counter
        self._decision_count = self._count_existing()
        
        logger.info(f"AuditLogger initialized. Model version: {self.model_version}")
    
    def _get_git_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.log_dir.parent
            )
            return result.stdout.strip() or "unknown"
        except:
            return "unknown"
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _init_csv(self):
        """Initialize CSV with headers if needed."""
        if not self.audit_log_path.exists():
            with open(self.audit_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(AuditRecord.__annotations__.keys()))
                writer.writeheader()
    
    def _count_existing(self) -> int:
        """Count existing audit records."""
        if not self.audit_log_path.exists():
            return 0
        with open(self.audit_log_path, 'r') as f:
            return sum(1 for _ in f) - 1  # Subtract header
    
    def log_decision(
        self,
        vol_prob: float,
        trend_prob: float,
        liq_prob: float,
        lattice_state: str,
        regime_confidence: float,
        signal: str,
        signal_source: str,
        order_type: Optional[str] = None,
        order_qty: Optional[float] = None,
        order_notional: Optional[float] = None,
        order_id: Optional[str] = None,
        order_status: Optional[str] = None,
        fill_price: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> AuditRecord:
        """
        Log a complete trade decision.
        
        Returns:
            AuditRecord created
        """
        self._decision_count += 1
        
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision_id=f"D{self._decision_count:06d}",
            vol_prob=vol_prob,
            trend_prob=trend_prob,
            liq_prob=liq_prob,
            lattice_state=lattice_state,
            regime_confidence=regime_confidence,
            signal=signal,
            signal_source=signal_source,
            order_type=order_type,
            order_qty=order_qty,
            order_notional=order_notional,
            order_id=order_id,
            order_status=order_status,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            model_version=self.model_version,
            config_hash=self._get_config_hash(config or {}),
            notes=notes
        )
        
        # Write to CSV
        with open(self.audit_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(AuditRecord.__annotations__.keys()))
            writer.writerow(asdict(record))
        
        # Also append to JSON for easier analysis
        self._append_json(record)
        
        logger.info(f"Audit logged: {record.decision_id} - {signal} ({signal_source})")
        
        return record
    
    def _append_json(self, record: AuditRecord):
        """Append record to JSON log."""
        records = []
        if self.audit_json_path.exists():
            try:
                with open(self.audit_json_path, 'r') as f:
                    records = json.load(f)
            except:
                records = []
        
        records.append(asdict(record))
        
        # Keep last 1000 records in JSON
        if len(records) > 1000:
            records = records[-1000:]
        
        with open(self.audit_json_path, 'w') as f:
            json.dump(records, f, indent=2)
    
    def get_recent_decisions(self, n: int = 10) -> List[AuditRecord]:
        """Get n most recent decisions."""
        if not self.audit_json_path.exists():
            return []
        
        with open(self.audit_json_path, 'r') as f:
            records = json.load(f)
        
        return [AuditRecord(**r) for r in records[-n:]]
    
    def get_decision_by_id(self, decision_id: str) -> Optional[AuditRecord]:
        """Get a specific decision by ID."""
        if not self.audit_json_path.exists():
            return None
        
        with open(self.audit_json_path, 'r') as f:
            records = json.load(f)
        
        for r in records:
            if r['decision_id'] == decision_id:
                return AuditRecord(**r)
        
        return None
    
    def export_report(self, output_path: Optional[Path] = None) -> Path:
        """Export audit summary report."""
        output_path = output_path or self.log_dir.parent / "reports" / "audit_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.audit_json_path.exists():
            report = {"total_decisions": 0, "decisions": []}
        else:
            with open(self.audit_json_path, 'r') as f:
                records = json.load(f)
            
            # Summary statistics
            signals = {}
            sources = {}
            for r in records:
                signals[r['signal']] = signals.get(r['signal'], 0) + 1
                sources[r['signal_source']] = sources.get(r['signal_source'], 0) + 1
            
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_decisions": len(records),
                "signal_counts": signals,
                "source_counts": sources,
                "model_version": self.model_version,
                "recent_decisions": records[-20:]
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Audit report exported: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test audit logger
    audit = AuditLogger()
    
    # Log a test decision
    record = audit.log_decision(
        vol_prob=0.3,
        trend_prob=0.7,
        liq_prob=0.1,
        lattice_state="FAVORABLE",
        regime_confidence=0.6,
        signal="ENTER_LONG",
        signal_source="regime_model",
        order_type="market",
        order_notional=10000,
        notes="Test decision"
    )
    
    print(f"Logged: {record.decision_id}")
    
    # Get recent
    recent = audit.get_recent_decisions(5)
    print(f"Recent decisions: {len(recent)}")
    
    # Export report
    report_path = audit.export_report()
    print(f"Report: {report_path}")
