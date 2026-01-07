"""
Decay Detector

Early-warning signals for edge degradation (non-performance-based):
- Regime probability clustering (model uncertainty)
- Volatility classifier entropy collapse
- Regime duration drift vs training distribution
- Correlation drift with historical labels

These trigger INVESTIGATION, not trades.

NO STRATEGY CHANGES. OBSERVE ONLY.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DecayAlert:
    """A single decay alert."""
    timestamp: str
    alert_type: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    description: str


@dataclass  
class DecayStatus:
    """Overall decay detection status."""
    is_healthy: bool
    alerts: List[DecayAlert]
    entropy_score: float
    clustering_score: float
    duration_drift: float
    correlation_drift: float
    overall_decay_risk: float  # 0-1


class DecayDetector:
    """
    Detects early signs of edge decay.
    
    Usage:
        detector = DecayDetector()
        status = detector.check_all(regime_probs, regime_history)
        if not status.is_healthy:
            for alert in status.alerts:
                print(f"ALERT: {alert.description}")
    """
    
    def __init__(
        self,
        entropy_threshold: float = 0.3,  # Min entropy (0 = certain = bad)
        clustering_threshold: float = 0.8,  # Max prob clustering
        duration_drift_threshold: float = 0.3,  # Relative drift allowed
        correlation_threshold: float = 0.5,  # Min correlation with training
        reports_dir: Optional[Path] = None
    ):
        self.entropy_threshold = entropy_threshold
        self.clustering_threshold = clustering_threshold
        self.duration_drift_threshold = duration_drift_threshold
        self.correlation_threshold = correlation_threshold
        
        project_root = Path(__file__).parent.parent.parent
        self.reports_dir = reports_dir or project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DecayDetector initialized")
    
    def calculate_entropy(self, probabilities: pd.DataFrame) -> float:
        """
        Calculate average prediction entropy.
        
        Low entropy = model too certain = potential overfit/decay
        
        Args:
            probabilities: DataFrame with probability columns (e.g., vol_prob, trend_prob)
        """
        if probabilities.empty:
            return 1.0  # Unknown
        
        entropies = []
        for col in probabilities.columns:
            probs = probabilities[col].dropna().values
            # Clip to avoid log(0)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            # Binary entropy
            entropy = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
            entropies.append(np.mean(entropy))
        
        return float(np.mean(entropies)) if entropies else 1.0
    
    def check_probability_clustering(self, probabilities: pd.DataFrame) -> Tuple[float, bool]:
        """
        Check if predictions are clustering unnaturally.
        
        All probabilities near 0 or 1 = suspicious
        
        Returns:
            (clustering_score, is_problematic)
        """
        if probabilities.empty:
            return 0.0, False
        
        # Check how many predictions are extreme
        extreme_count = 0
        total_count = 0
        
        for col in probabilities.columns:
            probs = probabilities[col].dropna().values
            # Count predictions < 0.1 or > 0.9
            extreme = np.sum((probs < 0.1) | (probs > 0.9))
            extreme_count += extreme
            total_count += len(probs)
        
        if total_count == 0:
            return 0.0, False
        
        clustering_score = extreme_count / total_count
        is_problematic = clustering_score > self.clustering_threshold
        
        return clustering_score, is_problematic
    
    def check_regime_duration_drift(
        self,
        current_durations: Dict[str, float],  # {"HOSTILE": avg_days, ...}
        training_durations: Dict[str, float]
    ) -> Tuple[float, bool]:
        """
        Check if regime durations have drifted from training distribution.
        
        Returns:
            (drift_score, is_problematic)
        """
        if not current_durations or not training_durations:
            return 0.0, False
        
        drifts = []
        for regime in current_durations:
            if regime in training_durations and training_durations[regime] > 0:
                relative_drift = abs(current_durations[regime] - training_durations[regime]) / training_durations[regime]
                drifts.append(relative_drift)
        
        if not drifts:
            return 0.0, False
        
        avg_drift = np.mean(drifts)
        is_problematic = avg_drift > self.duration_drift_threshold
        
        return float(avg_drift), is_problematic
    
    def check_correlation_drift(
        self,
        current_labels: pd.Series,
        historical_labels: pd.Series
    ) -> Tuple[float, bool]:
        """
        Check correlation between current and historical regime patterns.
        
        Low correlation = structural market change
        
        Returns:
            (correlation, is_problematic)
        """
        if current_labels.empty or historical_labels.empty:
            return 1.0, False
        
        # Align and compare
        common_len = min(len(current_labels), len(historical_labels))
        current = current_labels.iloc[-common_len:].values
        historical = historical_labels.iloc[-common_len:].values
        
        # Convert to numeric if categorical
        if current.dtype == object:
            unique_vals = np.unique(np.concatenate([current, historical]))
            mapping = {v: i for i, v in enumerate(unique_vals)}
            current = np.array([mapping.get(v, 0) for v in current])
            historical = np.array([mapping.get(v, 0) for v in historical])
        
        try:
            corr, _ = stats.spearmanr(current, historical)
            if np.isnan(corr):
                corr = 0.0
        except:
            corr = 0.0
        
        is_problematic = corr < self.correlation_threshold
        
        return float(corr), is_problematic
    
    def check_all(
        self,
        regime_probabilities: pd.DataFrame,
        current_regime_durations: Dict[str, float],
        training_regime_durations: Dict[str, float],
        current_labels: pd.Series,
        historical_labels: pd.Series
    ) -> DecayStatus:
        """
        Run all decay detection checks.
        
        Returns:
            DecayStatus with all alerts and scores
        """
        alerts: List[DecayAlert] = []
        now = datetime.now(timezone.utc).isoformat()
        
        # 1. Entropy check
        entropy = self.calculate_entropy(regime_probabilities)
        if entropy < self.entropy_threshold:
            alerts.append(DecayAlert(
                timestamp=now,
                alert_type="entropy_collapse",
                severity=AlertSeverity.WARNING,
                metric_name="prediction_entropy",
                current_value=entropy,
                threshold=self.entropy_threshold,
                description=f"Entropy collapse: {entropy:.3f} < {self.entropy_threshold}. Model may be overfitting."
            ))
        
        # 2. Clustering check
        clustering, cluster_problem = self.check_probability_clustering(regime_probabilities)
        if cluster_problem:
            alerts.append(DecayAlert(
                timestamp=now,
                alert_type="probability_clustering",
                severity=AlertSeverity.WARNING,
                metric_name="extreme_prediction_rate",
                current_value=clustering,
                threshold=self.clustering_threshold,
                description=f"Probability clustering: {clustering:.1%} extreme predictions."
            ))
        
        # 3. Duration drift check
        duration_drift, duration_problem = self.check_regime_duration_drift(
            current_regime_durations, training_regime_durations
        )
        if duration_problem:
            alerts.append(DecayAlert(
                timestamp=now,
                alert_type="duration_drift",
                severity=AlertSeverity.CRITICAL,
                metric_name="regime_duration_drift",
                current_value=duration_drift,
                threshold=self.duration_drift_threshold,
                description=f"Regime duration drift: {duration_drift:.1%} from training. Market may have changed."
            ))
        
        # 4. Correlation drift check
        correlation, corr_problem = self.check_correlation_drift(current_labels, historical_labels)
        if corr_problem:
            alerts.append(DecayAlert(
                timestamp=now,
                alert_type="correlation_drift",
                severity=AlertSeverity.CRITICAL,
                metric_name="label_correlation",
                current_value=correlation,
                threshold=self.correlation_threshold,
                description=f"Correlation drift: {correlation:.2f} with historical patterns."
            ))
        
        # Overall decay risk
        risk_factors = [
            1 - min(entropy / 1.0, 1.0),  # Low entropy = high risk
            clustering,
            duration_drift,
            1 - max(correlation, 0)  # Low correlation = high risk
        ]
        overall_risk = np.mean(risk_factors)
        
        is_healthy = len(alerts) == 0
        
        return DecayStatus(
            is_healthy=is_healthy,
            alerts=alerts,
            entropy_score=entropy,
            clustering_score=clustering,
            duration_drift=duration_drift,
            correlation_drift=correlation,
            overall_decay_risk=overall_risk
        )
    
    def generate_report(self, status: DecayStatus) -> str:
        """Generate markdown decay detection report."""
        
        health_emoji = "🟢" if status.is_healthy else ("🟡" if status.overall_decay_risk < 0.5 else "🔴")
        
        content = f"""# Decay Detection Report

**Generated**: {datetime.now(timezone.utc).isoformat()}

## Overall Status

{health_emoji} **{"HEALTHY" if status.is_healthy else "INVESTIGATION NEEDED"}**

**Decay Risk Score**: {status.overall_decay_risk:.1%}

---

## Metrics

| Check | Score | Status |
|-------|-------|--------|
| Entropy | {status.entropy_score:.3f} | {"✅" if status.entropy_score >= self.entropy_threshold else "⚠️"} |
| Clustering | {status.clustering_score:.1%} | {"✅" if status.clustering_score < self.clustering_threshold else "⚠️"} |
| Duration Drift | {status.duration_drift:.1%} | {"✅" if status.duration_drift < self.duration_drift_threshold else "⚠️"} |
| Correlation | {status.correlation_drift:.2f} | {"✅" if status.correlation_drift >= self.correlation_threshold else "⚠️"} |

---

## Alerts

"""
        if status.alerts:
            for alert in status.alerts:
                severity_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[alert.severity.value]
                content += f"### {severity_emoji} {alert.alert_type.upper()}\n\n"
                content += f"**{alert.description}**\n\n"
                content += f"- Current: {alert.current_value:.3f}\n"
                content += f"- Threshold: {alert.threshold:.3f}\n\n"
        else:
            content += "No alerts. System healthy.\n"
        
        content += """
---

## Recommended Actions

If alerts are present:
1. **Do NOT change strategy parameters**
2. Review recent market structure
3. Compare regime distributions to training period
4. Document findings in stress log
"""
        return content
    
    def save_report(self, status: DecayStatus) -> Path:
        """Save decay report to file."""
        content = self.generate_report(status)
        path = self.reports_dir / f"decay_detection_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(path, 'w') as f:
            f.write(content)
        
        logger.info(f"Decay report saved: {path}")
        return path


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    
    # Simulate probabilities
    probs = pd.DataFrame({
        "vol_prob": np.random.beta(2, 5, len(dates)),
        "trend_prob": np.random.beta(3, 3, len(dates)),
        "liq_prob": np.random.beta(1, 10, len(dates))
    }, index=dates)
    
    # Simulate durations
    training_durations = {"HOSTILE": 5.0, "NEUTRAL": 10.0, "FAVORABLE": 15.0}
    current_durations = {"HOSTILE": 7.0, "NEUTRAL": 8.0, "FAVORABLE": 12.0}
    
    # Simulate labels
    current = pd.Series(np.random.choice(["H", "N", "F"], len(dates)), index=dates)
    historical = pd.Series(np.random.choice(["H", "N", "F"], len(dates)), index=dates)
    
    detector = DecayDetector()
    status = detector.check_all(
        probs, current_durations, training_durations, current, historical
    )
    
    print(f"Healthy: {status.is_healthy}")
    print(f"Decay Risk: {status.overall_decay_risk:.1%}")
    print(f"Alerts: {len(status.alerts)}")
    
    detector.save_report(status)
