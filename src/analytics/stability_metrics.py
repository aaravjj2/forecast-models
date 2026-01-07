"""
Stability Metrics

Rolling performance stability tracking:
- 30 / 60 / 90 day Sharpe
- Drawdown slope
- Exposure utilization
- Trade frequency stability

NO STRATEGY CHANGES. OBSERVE ONLY.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StabilitySnapshot:
    """Stability metrics at a point in time."""
    date: str
    
    # Rolling Sharpe
    sharpe_30d: float
    sharpe_60d: float
    sharpe_90d: float
    
    # Drawdown
    current_drawdown: float
    drawdown_slope_30d: float  # Rate of change
    
    # Exposure
    exposure_utilization: float  # % of time exposed
    
    # Trade frequency
    trades_30d: int
    trade_frequency_stability: float  # Variance vs expected


@dataclass
class StabilityReport:
    """Stability analysis over a period."""
    period_start: str
    period_end: str
    
    # Current state
    current: StabilitySnapshot
    
    # Trends
    sharpe_trend: str  # "stable", "improving", "degrading"
    drawdown_trend: str
    exposure_trend: str
    
    # Warnings
    warnings: List[str]
    
    # Overall stability score
    stability_score: float  # 0-100


class StabilityMetrics:
    """
    Tracks rolling stability metrics.
    
    Usage:
        stability = StabilityMetrics()
        report = stability.analyze(returns, positions)
    """
    
    def __init__(
        self,
        expected_trade_frequency: float = 2.0,  # Trades per week
        reports_dir: Optional[Path] = None
    ):
        self.expected_trade_frequency = expected_trade_frequency
        
        project_root = Path(__file__).parent.parent.parent
        self.reports_dir = reports_dir or project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("StabilityMetrics initialized")
    
    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return rolling_sharpe.fillna(0)
    
    def calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown
    
    def calculate_drawdown_slope(
        self,
        drawdown: pd.Series,
        window: int = 30
    ) -> float:
        """Calculate rate of change in drawdown."""
        if len(drawdown) < window:
            return 0.0
        
        recent = drawdown.iloc[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent.values, 1)[0]
        
        return float(slope)
    
    def calculate_exposure_utilization(
        self,
        positions: pd.Series,
        window: int = 30
    ) -> float:
        """Calculate % of time exposed in window."""
        if len(positions) < window:
            return positions.mean()
        
        recent = positions.iloc[-window:]
        return float(recent.mean())
    
    def calculate_trade_frequency_stability(
        self,
        positions: pd.Series,
        window: int = 30
    ) -> tuple:
        """Calculate trade count and stability."""
        if len(positions) < window:
            recent = positions
        else:
            recent = positions.iloc[-window:]
        
        # Count position changes
        changes = (recent.diff() != 0).sum()
        
        # Expected trades in window
        weeks_in_window = window / 5
        expected = self.expected_trade_frequency * weeks_in_window
        
        # Stability = 1 - abs(deviation) / expected
        if expected > 0:
            deviation = abs(changes - expected) / expected
            stability = max(0, 1 - deviation)
        else:
            stability = 1.0
        
        return int(changes), float(stability)
    
    def get_snapshot(
        self,
        returns: pd.Series,
        positions: pd.Series,
        equity: pd.Series
    ) -> StabilitySnapshot:
        """Get current stability snapshot."""
        now = returns.index[-1]
        date_str = str(now.date()) if hasattr(now, 'date') else str(now)
        
        # Rolling Sharpe
        sharpe_30 = self.calculate_rolling_sharpe(returns, 30).iloc[-1]
        sharpe_60 = self.calculate_rolling_sharpe(returns, 60).iloc[-1]
        sharpe_90 = self.calculate_rolling_sharpe(returns, 90).iloc[-1]
        
        # Drawdown
        dd = self.calculate_drawdown(equity)
        current_dd = dd.iloc[-1]
        dd_slope = self.calculate_drawdown_slope(dd)
        
        # Exposure
        exposure = self.calculate_exposure_utilization(positions)
        
        # Trade frequency
        trades, stability = self.calculate_trade_frequency_stability(positions)
        
        return StabilitySnapshot(
            date=date_str,
            sharpe_30d=float(sharpe_30),
            sharpe_60d=float(sharpe_60),
            sharpe_90d=float(sharpe_90),
            current_drawdown=float(current_dd) * 100,
            drawdown_slope_30d=float(dd_slope) * 100,
            exposure_utilization=float(exposure) * 100,
            trades_30d=trades,
            trade_frequency_stability=stability
        )
    
    def analyze(
        self,
        returns: pd.Series,
        positions: pd.Series,
        equity: Optional[pd.Series] = None
    ) -> StabilityReport:
        """Run full stability analysis."""
        if equity is None:
            equity = (1 + returns).cumprod()
        
        snapshot = self.get_snapshot(returns, positions, equity)
        
        # Trend analysis
        warnings = []
        
        # Sharpe trend
        if snapshot.sharpe_30d < snapshot.sharpe_90d - 0.3:
            sharpe_trend = "degrading"
            warnings.append("Sharpe ratio declining (30d < 90d by >0.3)")
        elif snapshot.sharpe_30d > snapshot.sharpe_90d + 0.3:
            sharpe_trend = "improving"
        else:
            sharpe_trend = "stable"
        
        # Drawdown trend
        if snapshot.drawdown_slope_30d < -0.1:  # Getting worse
            drawdown_trend = "degrading"
            warnings.append(f"Drawdown worsening (slope: {snapshot.drawdown_slope_30d:.2f}%/day)")
        elif snapshot.drawdown_slope_30d > 0.05:
            drawdown_trend = "recovering"
        else:
            drawdown_trend = "stable"
        
        # Exposure trend
        if snapshot.exposure_utilization < 50:
            exposure_trend = "underutilized"
            warnings.append("Exposure below 50% - strategy mostly flat")
        elif snapshot.exposure_utilization > 90:
            exposure_trend = "overutilized"
            warnings.append("Exposure above 90% - little crash protection")
        else:
            exposure_trend = "balanced"
        
        # Trade frequency warning
        if snapshot.trade_frequency_stability < 0.5:
            warnings.append("Trade frequency unstable vs expected")
        
        # Overall stability score
        sharpe_score = min(40, max(0, snapshot.sharpe_30d * 20))
        dd_score = max(0, 30 + snapshot.current_drawdown)  # Less DD = higher score
        exposure_score = 30 - abs(snapshot.exposure_utilization - 70) * 0.3
        
        stability_score = sharpe_score + dd_score + exposure_score
        
        start = str(returns.index[0].date()) if hasattr(returns.index[0], 'date') else str(returns.index[0])
        end = str(returns.index[-1].date()) if hasattr(returns.index[-1], 'date') else str(returns.index[-1])
        
        return StabilityReport(
            period_start=start,
            period_end=end,
            current=snapshot,
            sharpe_trend=sharpe_trend,
            drawdown_trend=drawdown_trend,
            exposure_trend=exposure_trend,
            warnings=warnings,
            stability_score=stability_score
        )
    
    def generate_scorecard(self, report: StabilityReport) -> str:
        """Generate stability scorecard."""
        snapshot = report.current
        
        # Score color
        if report.stability_score >= 70:
            score_emoji = "🟢"
        elif report.stability_score >= 40:
            score_emoji = "🟡"
        else:
            score_emoji = "🔴"
        
        content = f"""# Stability Scorecard

**Period**: {report.period_start} to {report.period_end}
**Score**: {score_emoji} **{report.stability_score:.0f}/100**

---

## Rolling Sharpe Ratio

| Window | Sharpe | Trend |
|--------|--------|-------|
| 30 day | {snapshot.sharpe_30d:.2f} | |
| 60 day | {snapshot.sharpe_60d:.2f} | |
| 90 day | {snapshot.sharpe_90d:.2f} | **{report.sharpe_trend}** |

---

## Drawdown

| Metric | Value |
|--------|-------|
| Current DD | {snapshot.current_drawdown:.2f}% |
| DD Slope (30d) | {snapshot.drawdown_slope_30d:.3f}%/day |
| Trend | **{report.drawdown_trend}** |

---

## Exposure

| Metric | Value |
|--------|-------|
| Utilization | {snapshot.exposure_utilization:.1f}% |
| Trades (30d) | {snapshot.trades_30d} |
| Frequency Stability | {snapshot.trade_frequency_stability:.1%} |
| Trend | **{report.exposure_trend}** |

---

## Warnings

"""
        if report.warnings:
            for w in report.warnings:
                content += f"- ⚠️ {w}\n"
        else:
            content += "No warnings. System stable.\n"
        
        content += f"\n*Generated: {datetime.now(timezone.utc).isoformat()}*\n"
        
        return content
    
    def save_scorecard(self, report: StabilityReport) -> Path:
        """Save scorecard to file."""
        content = self.generate_scorecard(report)
        path = self.reports_dir / f"stability_scorecard_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(path, 'w') as f:
            f.write(content)
        
        logger.info(f"Stability scorecard saved: {path}")
        return path


if __name__ == "__main__":
    # Example
    dates = pd.date_range("2025-01-01", periods=120, freq="B")
    np.random.seed(42)
    
    returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
    positions = pd.Series(np.random.choice([0, 1], len(dates), p=[0.3, 0.7]), index=dates)
    equity = (1 + returns * positions).cumprod() * 100000
    
    stability = StabilityMetrics()
    report = stability.analyze(returns, positions, equity)
    
    print(f"Stability Score: {report.stability_score:.0f}/100")
    print(f"Sharpe Trend: {report.sharpe_trend}")
    print(f"Warnings: {report.warnings}")
    
    stability.save_scorecard(report)
