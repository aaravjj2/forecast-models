"""
Regime Persistence Analysis

Measures regime behavior over time:
- Average Risk-On / Risk-Off duration
- Transition probabilities (Markov matrix)
- Comparison vs training distribution
- Detection of structural market changes

NO STRATEGY CHANGES. OBSERVE ONLY.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegimePersistence:
    """Regime persistence statistics."""
    regime: str
    avg_duration_days: float
    std_duration_days: float
    min_duration: int
    max_duration: int
    occurrence_count: int
    total_days: int


@dataclass
class TransitionMatrix:
    """Markov transition probabilities."""
    states: List[str]
    matrix: Dict[str, Dict[str, float]]  # P(next | current)


@dataclass
class PersistenceReport:
    """Complete regime persistence report."""
    period_start: str
    period_end: str
    total_days: int
    
    # Per-regime stats
    regime_stats: Dict[str, RegimePersistence]
    
    # Transitions
    transition_matrix: TransitionMatrix
    
    # Drift detection
    training_comparison: Dict[str, float]  # regime: drift_pct
    significant_drift: bool


class RegimePersistenceAnalyzer:
    """
    Analyzes regime persistence and transitions.
    
    Usage:
        analyzer = RegimePersistenceAnalyzer()
        report = analyzer.analyze(regime_series, training_stats)
    """
    
    def __init__(self, reports_dir: Optional[Path] = None):
        project_root = Path(__file__).parent.parent.parent
        self.reports_dir = reports_dir or project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RegimePersistenceAnalyzer initialized")
    
    def calculate_durations(self, regimes: pd.Series) -> Dict[str, RegimePersistence]:
        """Calculate duration statistics for each regime."""
        if regimes.empty:
            return {}
        
        # Find regime runs
        regime_runs = []
        current_regime = regimes.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_duration += 1
            else:
                regime_runs.append((current_regime, current_duration))
                current_regime = regimes.iloc[i]
                current_duration = 1
        regime_runs.append((current_regime, current_duration))
        
        # Calculate stats per regime
        unique_regimes = regimes.unique()
        stats = {}
        
        for regime in unique_regimes:
            durations = [d for r, d in regime_runs if r == regime]
            if durations:
                stats[str(regime)] = RegimePersistence(
                    regime=str(regime),
                    avg_duration_days=float(np.mean(durations)),
                    std_duration_days=float(np.std(durations)) if len(durations) > 1 else 0.0,
                    min_duration=int(np.min(durations)),
                    max_duration=int(np.max(durations)),
                    occurrence_count=len(durations),
                    total_days=sum(durations)
                )
        
        return stats
    
    def calculate_transitions(self, regimes: pd.Series) -> TransitionMatrix:
        """Calculate Markov transition probabilities."""
        if len(regimes) < 2:
            return TransitionMatrix(states=[], matrix={})
        
        unique_regimes = list(regimes.unique())
        
        # Count transitions
        transition_counts = {r: {r2: 0 for r2 in unique_regimes} for r in unique_regimes}
        
        for i in range(1, len(regimes)):
            prev = regimes.iloc[i-1]
            curr = regimes.iloc[i]
            transition_counts[prev][curr] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for r in unique_regimes:
            total = sum(transition_counts[r].values())
            transition_probs[str(r)] = {
                str(r2): (transition_counts[r][r2] / total if total > 0 else 0)
                for r2 in unique_regimes
            }
        
        return TransitionMatrix(
            states=[str(r) for r in unique_regimes],
            matrix=transition_probs
        )
    
    def compare_to_training(
        self,
        current_stats: Dict[str, RegimePersistence],
        training_stats: Dict[str, float]  # regime: avg_duration
    ) -> Tuple[Dict[str, float], bool]:
        """Compare current durations to training baseline."""
        comparison = {}
        significant = False
        
        for regime, stats in current_stats.items():
            if regime in training_stats and training_stats[regime] > 0:
                drift = (stats.avg_duration_days - training_stats[regime]) / training_stats[regime]
                comparison[regime] = drift
                if abs(drift) > 0.3:  # 30% drift = significant
                    significant = True
        
        return comparison, significant
    
    def analyze(
        self,
        regimes: pd.Series,
        training_durations: Optional[Dict[str, float]] = None
    ) -> PersistenceReport:
        """Run full persistence analysis."""
        stats = self.calculate_durations(regimes)
        transitions = self.calculate_transitions(regimes)
        
        if training_durations:
            comparison, drift = self.compare_to_training(stats, training_durations)
        else:
            comparison, drift = {}, False
        
        start = str(regimes.index[0].date()) if hasattr(regimes.index[0], 'date') else str(regimes.index[0])
        end = str(regimes.index[-1].date()) if hasattr(regimes.index[-1], 'date') else str(regimes.index[-1])
        
        return PersistenceReport(
            period_start=start,
            period_end=end,
            total_days=len(regimes),
            regime_stats=stats,
            transition_matrix=transitions,
            training_comparison=comparison,
            significant_drift=drift
        )
    
    def generate_report(self, report: PersistenceReport) -> str:
        """Generate markdown report."""
        content = f"""# Regime Persistence Analysis

**Period**: {report.period_start} to {report.period_end}
**Total Days**: {report.total_days}

## Duration Statistics

| Regime | Avg Days | Std | Min | Max | Occurrences |
|--------|----------|-----|-----|-----|-------------|
"""
        for regime, stats in report.regime_stats.items():
            content += f"| {regime} | {stats.avg_duration_days:.1f} | {stats.std_duration_days:.1f} | {stats.min_duration} | {stats.max_duration} | {stats.occurrence_count} |\n"
        
        content += """
## Transition Matrix

"""
        if report.transition_matrix.states:
            headers = "| From \\ To | " + " | ".join(report.transition_matrix.states) + " |\n"
            separator = "|" + "|".join(["---"] * (len(report.transition_matrix.states) + 1)) + "|\n"
            content += headers + separator
            
            for state in report.transition_matrix.states:
                row = f"| **{state}** |"
                for next_state in report.transition_matrix.states:
                    prob = report.transition_matrix.matrix.get(state, {}).get(next_state, 0)
                    row += f" {prob:.1%} |"
                content += row + "\n"
        
        if report.training_comparison:
            content += """
## Training Comparison

| Regime | Drift from Training |
|--------|---------------------|
"""
            for regime, drift in report.training_comparison.items():
                emoji = "⚠️" if abs(drift) > 0.3 else "✅"
                content += f"| {regime} | {drift:+.1%} {emoji} |\n"
            
            if report.significant_drift:
                content += "\n> [!WARNING]\n> Significant regime duration drift detected.\n"
        
        content += f"\n*Generated: {datetime.now(timezone.utc).isoformat()}*\n"
        
        return content
    
    def save_report(self, report: PersistenceReport) -> Path:
        """Save report to file."""
        content = self.generate_report(report)
        path = self.reports_dir / f"regime_persistence_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(path, 'w') as f:
            f.write(content)
        
        logger.info(f"Persistence report saved: {path}")
        return path


if __name__ == "__main__":
    # Example
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    
    # Simulate regimes with persistence
    regimes = []
    current = "NEUTRAL"
    for _ in range(100):
        if np.random.random() < 0.8:  # 80% stay same
            regimes.append(current)
        else:
            current = np.random.choice(["HOSTILE", "NEUTRAL", "FAVORABLE"])
            regimes.append(current)
    
    regime_series = pd.Series(regimes, index=dates)
    
    analyzer = RegimePersistenceAnalyzer()
    report = analyzer.analyze(
        regime_series,
        training_durations={"HOSTILE": 3.0, "NEUTRAL": 8.0, "FAVORABLE": 10.0}
    )
    
    print(f"Significant Drift: {report.significant_drift}")
    for regime, stats in report.regime_stats.items():
        print(f"{regime}: avg={stats.avg_duration_days:.1f} days")
    
    analyzer.save_report(report)
