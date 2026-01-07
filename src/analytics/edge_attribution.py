"""
Edge Attribution

Decomposes daily PnL into component contributions:
- Market Beta (exposure vs flat)
- Volatility Avoidance (saved by being flat during Risk-Off)
- Timing Contribution (entry/exit quality)
- Cost Drag (commissions + spreads)
- Slippage Impact (actual vs expected fills)

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
class DailyAttribution:
    """Attribution breakdown for a single day."""
    date: str
    total_pnl: float
    
    # Components
    market_beta: float  # Return from being exposed
    vol_avoidance: float  # Return saved by being flat during Risk-Off
    timing_contribution: float  # Entry/exit quality vs market
    cost_drag: float  # Explicit costs (commissions)
    slippage_impact: float  # Actual vs expected
    
    # Context
    was_exposed: bool
    regime_state: str  # FAVORABLE, NEUTRAL, HOSTILE
    market_return: float  # What market did that day


@dataclass
class AttributionSummary:
    """Summary of attribution over a period."""
    period_start: str
    period_end: str
    total_days: int
    
    # Totals
    total_pnl: float
    total_market_beta: float
    total_vol_avoidance: float
    total_timing: float
    total_costs: float
    total_slippage: float
    
    # Percentages
    pct_from_beta: float
    pct_from_vol_avoidance: float
    pct_from_timing: float
    pct_cost_drag: float
    
    # Stats
    exposed_days: int
    flat_days: int
    risk_off_days: int
    
    # True edge source
    primary_edge_source: str


class EdgeAttribution:
    """
    PnL decomposition engine.
    
    Usage:
        attrib = EdgeAttribution()
        daily = attrib.calculate_daily(equity_curve, positions, market_data)
        summary = attrib.summarize(daily)
    """
    
    def __init__(
        self,
        cost_bps: float = 10.0,  # Expected cost per trade in bps
        logs_dir: Optional[Path] = None
    ):
        self.cost_bps = cost_bps
        project_root = Path(__file__).parent.parent.parent
        self.logs_dir = logs_dir or project_root / "logs"
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("EdgeAttribution initialized")
    
    def calculate_daily(
        self,
        equity_curve: pd.Series,
        positions: pd.Series,  # 1 = exposed, 0 = flat
        market_returns: pd.Series,
        regime_states: pd.Series,
        fill_prices: Optional[pd.Series] = None,
        expected_prices: Optional[pd.Series] = None
    ) -> List[DailyAttribution]:
        """
        Calculate daily attribution breakdown.
        
        Args:
            equity_curve: Daily equity values (indexed by date)
            positions: Binary position (1 = long, 0 = flat)
            market_returns: Daily market returns
            regime_states: Regime state per day
            fill_prices: Actual fill prices (optional)
            expected_prices: Expected fill prices (optional)
            
        Returns:
            List of DailyAttribution for each day
        """
        # Align all inputs
        common_idx = equity_curve.index.intersection(positions.index).intersection(market_returns.index)
        
        equity = equity_curve.loc[common_idx]
        pos = positions.loc[common_idx]
        mkt_ret = market_returns.loc[common_idx]
        regimes = regime_states.reindex(common_idx).fillna("UNKNOWN")
        
        # Calculate daily returns
        daily_returns = equity.pct_change().fillna(0)
        
        results = []
        prev_pos = 0
        
        for i, date in enumerate(common_idx):
            if i == 0:
                prev_pos = pos.iloc[0]
                continue
            
            date_str = str(date.date()) if hasattr(date, 'date') else str(date)
            total_pnl = float(daily_returns.iloc[i]) * 100  # As percentage
            market_ret_today = float(mkt_ret.iloc[i]) * 100
            is_exposed = bool(pos.iloc[i] == 1)
            regime = str(regimes.iloc[i])
            
            # Position change
            position_changed = pos.iloc[i] != prev_pos
            
            # 1. Market Beta: What we got from being exposed
            if is_exposed:
                market_beta = market_ret_today
            else:
                market_beta = 0.0
            
            # 2. Volatility Avoidance: What we saved by being flat during Risk-Off
            if not is_exposed and regime == "HOSTILE":
                # We avoided this loss
                vol_avoidance = -market_ret_today if market_ret_today < 0 else 0.0
            else:
                vol_avoidance = 0.0
            
            # 3. Timing Contribution: Difference from pure beta
            # If we were exposed, how did our timing compare to holding?
            if is_exposed:
                timing_contribution = total_pnl - market_beta
            else:
                timing_contribution = 0.0
            
            # 4. Cost Drag: Explicit costs from trading
            if position_changed:
                cost_drag = -self.cost_bps / 100  # Convert bps to %
            else:
                cost_drag = 0.0
            
            # 5. Slippage Impact: Actual vs expected
            slippage_impact = 0.0
            if fill_prices is not None and expected_prices is not None:
                if date in fill_prices.index and date in expected_prices.index:
                    actual = fill_prices.loc[date]
                    expected = expected_prices.loc[date]
                    if expected > 0:
                        slippage_impact = ((actual - expected) / expected) * 100
            
            results.append(DailyAttribution(
                date=date_str,
                total_pnl=total_pnl,
                market_beta=market_beta,
                vol_avoidance=vol_avoidance,
                timing_contribution=timing_contribution,
                cost_drag=cost_drag,
                slippage_impact=slippage_impact,
                was_exposed=is_exposed,
                regime_state=regime,
                market_return=market_ret_today
            ))
            
            prev_pos = pos.iloc[i]
        
        return results
    
    def summarize(self, daily: List[DailyAttribution]) -> AttributionSummary:
        """Summarize attribution over the period."""
        if not daily:
            return AttributionSummary(
                period_start="", period_end="", total_days=0,
                total_pnl=0, total_market_beta=0, total_vol_avoidance=0,
                total_timing=0, total_costs=0, total_slippage=0,
                pct_from_beta=0, pct_from_vol_avoidance=0,
                pct_from_timing=0, pct_cost_drag=0,
                exposed_days=0, flat_days=0, risk_off_days=0,
                primary_edge_source="unknown"
            )
        
        total_pnl = sum(d.total_pnl for d in daily)
        total_beta = sum(d.market_beta for d in daily)
        total_vol = sum(d.vol_avoidance for d in daily)
        total_timing = sum(d.timing_contribution for d in daily)
        total_costs = sum(d.cost_drag for d in daily)
        total_slippage = sum(d.slippage_impact for d in daily)
        
        exposed_days = sum(1 for d in daily if d.was_exposed)
        flat_days = len(daily) - exposed_days
        risk_off_days = sum(1 for d in daily if d.regime_state == "HOSTILE")
        
        # Calculate percentages
        total_positive = abs(total_beta) + abs(total_vol) + abs(total_timing)
        if total_positive > 0:
            pct_beta = (abs(total_beta) / total_positive) * 100
            pct_vol = (abs(total_vol) / total_positive) * 100
            pct_timing = (abs(total_timing) / total_positive) * 100
        else:
            pct_beta = pct_vol = pct_timing = 0
        
        pct_cost = (abs(total_costs) / abs(total_pnl) * 100) if total_pnl != 0 else 0
        
        # Determine primary edge source
        sources = {
            "market_beta": total_beta,
            "vol_avoidance": total_vol,
            "timing": total_timing
        }
        primary_edge = max(sources, key=lambda k: abs(sources[k]))
        
        return AttributionSummary(
            period_start=daily[0].date,
            period_end=daily[-1].date,
            total_days=len(daily),
            total_pnl=total_pnl,
            total_market_beta=total_beta,
            total_vol_avoidance=total_vol,
            total_timing=total_timing,
            total_costs=total_costs,
            total_slippage=total_slippage,
            pct_from_beta=pct_beta,
            pct_from_vol_avoidance=pct_vol,
            pct_from_timing=pct_timing,
            pct_cost_drag=pct_cost,
            exposed_days=exposed_days,
            flat_days=flat_days,
            risk_off_days=risk_off_days,
            primary_edge_source=primary_edge
        )
    
    def generate_report(
        self,
        daily: List[DailyAttribution],
        summary: AttributionSummary
    ) -> Path:
        """Generate markdown attribution report."""
        report_path = self.reports_dir / f"edge_attribution_{datetime.now().strftime('%Y%m%d')}.md"
        
        content = f"""# Edge Attribution Report

**Period**: {summary.period_start} to {summary.period_end}
**Generated**: {datetime.now(timezone.utc).isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total Days | {summary.total_days} |
| Total PnL | {summary.total_pnl:.2f}% |
| Exposed Days | {summary.exposed_days} ({summary.exposed_days/summary.total_days*100:.1f}%) |
| Flat Days | {summary.flat_days} ({summary.flat_days/summary.total_days*100:.1f}%) |
| Risk-Off Days | {summary.risk_off_days} |

## Attribution Breakdown

| Component | Contribution | % of Total |
|-----------|-------------|------------|
| **Market Beta** | {summary.total_market_beta:.2f}% | {summary.pct_from_beta:.1f}% |
| **Volatility Avoidance** | {summary.total_vol_avoidance:.2f}% | {summary.pct_from_vol_avoidance:.1f}% |
| **Timing** | {summary.total_timing:.2f}% | {summary.pct_from_timing:.1f}% |
| **Cost Drag** | {summary.total_costs:.2f}% | -{summary.pct_cost_drag:.1f}% |
| **Slippage** | {summary.total_slippage:.2f}% | - |

## Primary Edge Source

**{summary.primary_edge_source.upper()}**

{"The edge comes from being exposed when the market rises." if summary.primary_edge_source == "market_beta" else ""}
{"The edge comes from avoiding losses during Risk-Off periods." if summary.primary_edge_source == "vol_avoidance" else ""}
{"The edge comes from superior entry/exit timing." if summary.primary_edge_source == "timing" else ""}

## Daily Breakdown (Last 10 Days)

| Date | PnL | Beta | Vol Avoid | Regime |
|------|-----|------|-----------|--------|
"""
        for d in daily[-10:]:
            content += f"| {d.date} | {d.total_pnl:.2f}% | {d.market_beta:.2f}% | {d.vol_avoidance:.2f}% | {d.regime_state} |\n"
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Attribution report saved: {report_path}")
        return report_path


if __name__ == "__main__":
    # Example usage with synthetic data
    import numpy as np
    
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    
    # Synthetic data
    market_returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
    positions = pd.Series(np.random.choice([0, 1], len(dates), p=[0.3, 0.7]), index=dates)
    equity = (1 + (market_returns * positions)).cumprod() * 100000
    regimes = pd.Series(np.random.choice(["FAVORABLE", "NEUTRAL", "HOSTILE"], len(dates), p=[0.5, 0.3, 0.2]), index=dates)
    
    attrib = EdgeAttribution()
    daily = attrib.calculate_daily(equity, positions, market_returns, regimes)
    summary = attrib.summarize(daily)
    
    print(f"Total PnL: {summary.total_pnl:.2f}%")
    print(f"Primary Edge: {summary.primary_edge_source}")
    print(f"Vol Avoidance: {summary.total_vol_avoidance:.2f}%")
    
    attrib.generate_report(daily, summary)
