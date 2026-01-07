"""
Edge Health Metrics

True KPIs that measure edge quality (not raw return):
- Regime Hit Rate (correct Risk-Off predictions)
- Crash Avoidance Delta (vs Buy & Hold)
- Drawdown Prevented (absolute & %)
- Opportunity Cost (missed upside)

NO STRATEGY CHANGES. OBSERVE ONLY.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeHealthSnapshot:
    """Edge health metrics for a period."""
    period_start: str
    period_end: str
    
    # Regime Hit Rate
    regime_hit_rate: float  # % of correct Risk-Off predictions
    risk_off_predicted: int
    risk_off_actual: int  # Days with actual negative returns
    
    # Crash Avoidance
    crash_avoidance_delta: float  # Return saved vs Buy & Hold
    crashes_avoided: int  # Days when flat and market dropped > 1%
    crash_damage_avoided: float  # Total % of drawdown avoided
    
    # Drawdown Prevention
    max_drawdown_strategy: float
    max_drawdown_buyhold: float
    drawdown_prevented_pct: float
    drawdown_prevented_abs: float  # In $ if provided
    
    # Opportunity Cost
    opportunity_cost: float  # Upside missed during flat periods
    missed_rallies: int  # Days flat when market rose > 1%
    
    # Overall Health Score (0-100)
    health_score: float


class EdgeHealth:
    """
    Tracks and reports edge health metrics.
    
    Usage:
        health = EdgeHealth()
        snapshot = health.calculate(positions, market_returns, regime_predictions)
        dashboard = health.generate_dashboard(snapshot)
    """
    
    def __init__(
        self,
        crash_threshold: float = -0.01,  # 1% drop = crash
        rally_threshold: float = 0.01,   # 1% gain = rally
        reports_dir: Optional[Path] = None
    ):
        self.crash_threshold = crash_threshold
        self.rally_threshold = rally_threshold
        
        project_root = Path(__file__).parent.parent.parent
        self.reports_dir = reports_dir or project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("EdgeHealth initialized")
    
    def calculate(
        self,
        positions: pd.Series,  # 1 = long, 0 = flat
        market_returns: pd.Series,
        regime_predictions: pd.Series,  # HOSTILE = predicted Risk-Off
        starting_capital: float = 100000
    ) -> EdgeHealthSnapshot:
        """
        Calculate edge health metrics.
        
        Args:
            positions: Binary position series
            market_returns: Daily market returns
            regime_predictions: Regime state predictions
            starting_capital: Starting capital for $ calculations
        """
        # Align data
        common_idx = positions.index.intersection(market_returns.index)
        pos = positions.loc[common_idx]
        mkt = market_returns.loc[common_idx]
        regimes = regime_predictions.reindex(common_idx).fillna("UNKNOWN")
        
        # Strategy returns
        strategy_returns = mkt * pos
        
        # 1. Regime Hit Rate
        # Risk-Off prediction is correct when market had negative return
        risk_off_mask = regimes == "HOSTILE"
        actual_negative = mkt < 0
        
        risk_off_predicted = risk_off_mask.sum()
        correct_risk_off = (risk_off_mask & actual_negative).sum()
        
        if risk_off_predicted > 0:
            regime_hit_rate = correct_risk_off / risk_off_predicted
        else:
            regime_hit_rate = 0.0
        
        # 2. Crash Avoidance
        crashes = mkt < self.crash_threshold
        flat_during_crash = (pos == 0) & crashes
        crashes_avoided = flat_during_crash.sum()
        crash_damage_avoided = (-mkt[flat_during_crash]).sum() if crashes_avoided > 0 else 0.0
        
        # Total crash avoidance delta
        buyhold_crash_damage = mkt[crashes].sum()
        strategy_crash_damage = strategy_returns[crashes].sum()
        crash_avoidance_delta = buyhold_crash_damage - strategy_crash_damage
        
        # 3. Drawdown Prevention
        # Strategy equity curve
        strategy_equity = (1 + strategy_returns).cumprod()
        strategy_peak = strategy_equity.cummax()
        strategy_dd = (strategy_equity - strategy_peak) / strategy_peak
        max_dd_strategy = strategy_dd.min()
        
        # Buy & Hold equity curve
        buyhold_equity = (1 + mkt).cumprod()
        buyhold_peak = buyhold_equity.cummax()
        buyhold_dd = (buyhold_equity - buyhold_peak) / buyhold_peak
        max_dd_buyhold = buyhold_dd.min()
        
        dd_prevented_pct = max_dd_buyhold - max_dd_strategy
        dd_prevented_abs = dd_prevented_pct * starting_capital
        
        # 4. Opportunity Cost
        rallies = mkt > self.rally_threshold
        flat_during_rally = (pos == 0) & rallies
        missed_rallies = flat_during_rally.sum()
        opportunity_cost = mkt[flat_during_rally].sum() if missed_rallies > 0 else 0.0
        
        # 5. Health Score (weighted composite)
        # Regime hit rate: 30%
        # Crash avoidance: 40%
        # DD prevention: 20%
        # Opportunity cost penalty: 10%
        
        hit_score = regime_hit_rate * 30
        crash_score = min(40, crash_avoidance_delta * 100 * 10)  # Cap at 40
        dd_score = min(20, dd_prevented_pct * 100)  # Cap at 20
        opp_penalty = min(10, opportunity_cost * 50)  # Penalty
        
        health_score = max(0, min(100, hit_score + crash_score + dd_score - opp_penalty))
        
        return EdgeHealthSnapshot(
            period_start=str(common_idx[0].date()) if hasattr(common_idx[0], 'date') else str(common_idx[0]),
            period_end=str(common_idx[-1].date()) if hasattr(common_idx[-1], 'date') else str(common_idx[-1]),
            regime_hit_rate=regime_hit_rate,
            risk_off_predicted=int(risk_off_predicted),
            risk_off_actual=int(actual_negative.sum()),
            crash_avoidance_delta=float(crash_avoidance_delta) * 100,
            crashes_avoided=int(crashes_avoided),
            crash_damage_avoided=float(crash_damage_avoided) * 100,
            max_drawdown_strategy=float(max_dd_strategy) * 100,
            max_drawdown_buyhold=float(max_dd_buyhold) * 100,
            drawdown_prevented_pct=float(dd_prevented_pct) * 100,
            drawdown_prevented_abs=float(dd_prevented_abs),
            opportunity_cost=float(opportunity_cost) * 100,
            missed_rallies=int(missed_rallies),
            health_score=health_score
        )
    
    def generate_dashboard(self, snapshot: EdgeHealthSnapshot) -> str:
        """Generate markdown dashboard content."""
        
        # Health score color
        if snapshot.health_score >= 70:
            health_emoji = "🟢"
        elif snapshot.health_score >= 40:
            health_emoji = "🟡"
        else:
            health_emoji = "🔴"
        
        content = f"""# Edge Health Dashboard

**Period**: {snapshot.period_start} to {snapshot.period_end}

## Overall Health Score

{health_emoji} **{snapshot.health_score:.0f}/100**

---

## Regime Prediction Quality

| Metric | Value |
|--------|-------|
| **Hit Rate** | {snapshot.regime_hit_rate:.1%} |
| Risk-Off Predictions | {snapshot.risk_off_predicted} days |
| Actual Negative Days | {snapshot.risk_off_actual} |

---

## Crash Avoidance (True KPI)

| Metric | Value |
|--------|-------|
| **Crash Avoidance Delta** | +{snapshot.crash_avoidance_delta:.2f}% |
| Crashes Avoided | {snapshot.crashes_avoided} |
| Damage Avoided | {snapshot.crash_damage_avoided:.2f}% |

---

## Drawdown Prevention

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Max Drawdown | {snapshot.max_drawdown_strategy:.2f}% | {snapshot.max_drawdown_buyhold:.2f}% |

**Prevented**: {snapshot.drawdown_prevented_pct:.2f}% (${snapshot.drawdown_prevented_abs:,.0f})

---

## Opportunity Cost

| Metric | Value |
|--------|-------|
| Missed Upside | {snapshot.opportunity_cost:.2f}% |
| Missed Rallies | {snapshot.missed_rallies} days |

---

*Generated: {datetime.now(timezone.utc).isoformat()}*
"""
        return content
    
    def save_dashboard(self, snapshot: EdgeHealthSnapshot) -> Path:
        """Save dashboard to file."""
        content = self.generate_dashboard(snapshot)
        path = self.reports_dir / f"edge_health_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(path, 'w') as f:
            f.write(content)
        
        logger.info(f"Edge health dashboard saved: {path}")
        return path


if __name__ == "__main__":
    # Example with synthetic data
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    
    market_returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
    # Simulate some crashes
    market_returns.iloc[10] = -0.025
    market_returns.iloc[30] = -0.020
    market_returns.iloc[45] = -0.030
    
    positions = pd.Series([1] * len(dates), index=dates)
    # Go flat on crash days
    positions.iloc[10] = 0
    positions.iloc[30] = 0
    positions.iloc[45] = 0
    
    regimes = pd.Series(["NEUTRAL"] * len(dates), index=dates)
    regimes.iloc[10] = "HOSTILE"
    regimes.iloc[30] = "HOSTILE"
    regimes.iloc[45] = "HOSTILE"
    
    health = EdgeHealth()
    snapshot = health.calculate(positions, market_returns, regimes)
    
    print(f"Health Score: {snapshot.health_score:.0f}/100")
    print(f"Crash Avoidance: +{snapshot.crash_avoidance_delta:.2f}%")
    print(f"Hit Rate: {snapshot.regime_hit_rate:.1%}")
    
    health.save_dashboard(snapshot)
