"""
Weekly Reconciliation

Compares simulated backtest results vs actual paper trading performance.
Identifies divergence between expected and realized metrics.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of weekly reconciliation."""
    period_start: str
    period_end: str
    
    # Backtest expectations
    expected_return: float
    expected_sharpe: float
    expected_drawdown: float
    expected_turnover: int
    
    # Actual results
    actual_return: float
    actual_sharpe: float
    actual_drawdown: float
    actual_turnover: int
    actual_slippage_mean: float
    
    # Divergence
    return_divergence: float
    sharpe_divergence: float
    drawdown_divergence: float
    turnover_divergence: float
    
    # Status
    status: str  # PASS, WARN, FAIL
    warnings: List[str]


class WeeklyReconciliation:
    """
    Weekly reconciliation engine.
    
    Compares paper trading results to backtest expectations.
    """
    
    def __init__(
        self,
        logs_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None
    ):
        """
        Initialize reconciliation engine.
        
        Args:
            logs_dir: Directory containing trading logs
            reports_dir: Directory for reconciliation reports
        """
        project_root = Path(__file__).parent.parent.parent
        self.logs_dir = logs_dir or project_root / "logs"
        self.reports_dir = reports_dir or project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("WeeklyReconciliation initialized")
    
    def load_paper_trades(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load paper trades for a period."""
        trades_path = self.logs_dir / "paper_trades.csv"
        
        if not trades_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(trades_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        return df[mask]
    
    def load_daily_pnl(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load daily PnL series for a period."""
        pnl_path = self.logs_dir / "daily_pnl.csv"
        
        if not pnl_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(pnl_path)
        df['date'] = pd.to_datetime(df['date'])
        
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        return df[mask]
    
    def calculate_actual_metrics(
        self,
        trades: pd.DataFrame,
        pnl: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate actual metrics from paper trading data."""
        if pnl.empty:
            return {
                'return': 0.0,
                'sharpe': 0.0,
                'drawdown': 0.0,
                'turnover': 0
            }
        
        # Return
        if 'equity' in pnl.columns:
            start_equity = pnl.iloc[0]['equity']
            end_equity = pnl.iloc[-1]['equity']
            total_return = (end_equity / start_equity - 1) * 100
        else:
            total_return = pnl['daily_return'].sum() if 'daily_return' in pnl.columns else 0
        
        # Sharpe (simplified)
        if 'daily_return' in pnl.columns and len(pnl) > 1:
            daily_returns = pnl['daily_return'].values
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe = 0.0
        
        # Drawdown
        if 'equity' in pnl.columns:
            equity = pnl['equity'].values
            peak = np.maximum.accumulate(equity)
            drawdown = ((peak - equity) / peak).max() * 100
        else:
            drawdown = 0.0
        
        # Turnover (count of actual trades)
        turnover = len(trades[trades['order_status'] != 'NO_ACTION']) if not trades.empty else 0
        
        return {
            'return': total_return,
            'sharpe': sharpe,
            'drawdown': drawdown,
            'turnover': turnover
        }
    
    def get_backtest_expectations(self) -> Dict[str, float]:
        """
        Get expected metrics from backtest.
        
        These are the "budget" values we expect from historical analysis.
        """
        # These should be updated based on backtest results
        # Currently using Phase 6 validation results for SPY-like behavior
        return {
            'return_per_year': 15.0,  # % per year (scaled for period)
            'sharpe': 1.0,
            'max_drawdown': 20.0,  # %
            'turnover_per_week': 2  # Expected trades per week
        }
    
    def reconcile(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ReconciliationResult:
        """
        Run weekly reconciliation.
        
        Args:
            start_date: Start of period (default: 7 days ago)
            end_date: End of period (default: today)
            
        Returns:
            ReconciliationResult with comparison metrics
        """
        # Default to last 7 days
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=7)
        
        logger.info(f"Reconciling {start_date.date()} to {end_date.date()}")
        
        # Load data
        trades = self.load_paper_trades(start_date, end_date)
        pnl = self.load_daily_pnl(start_date, end_date)
        
        # Calculate actual metrics
        actual = self.calculate_actual_metrics(trades, pnl)
        
        # Get expectations
        expected = self.get_backtest_expectations()
        
        # Scale expected return for the period
        days_in_period = (end_date - start_date).days
        expected_period_return = expected['return_per_year'] * (days_in_period / 365)
        expected_turnover = expected['turnover_per_week'] * (days_in_period / 7)
        
        # Calculate divergence
        return_div = actual['return'] - expected_period_return
        sharpe_div = actual['sharpe'] - expected['sharpe']
        dd_div = actual['drawdown'] - expected['max_drawdown']
        turnover_div = actual['turnover'] - expected_turnover
        
        # Check for warnings
        warnings = []
        status = "PASS"
        
        if abs(return_div) > 10:  # 10% divergence
            warnings.append(f"Return divergence: {return_div:.1f}%")
            status = "WARN"
        
        if sharpe_div < -0.5:  # Sharpe 0.5 lower
            warnings.append(f"Sharpe divergence: {sharpe_div:.2f}")
            status = "WARN"
        
        if dd_div > 10:  # Drawdown 10% worse
            warnings.append(f"Drawdown divergence: {dd_div:.1f}%")
            status = "FAIL"
        
        if abs(turnover_div) > 5:  # 5+ extra trades
            warnings.append(f"Turnover divergence: {turnover_div:.0f} trades")
            status = "WARN" if status != "FAIL" else status
        
        # Get slippage from fill tracker
        slippage_path = self.logs_dir / "fills.csv"
        if slippage_path.exists():
            fills = pd.read_csv(slippage_path)
            slippage_mean = fills['slippage_bps'].mean() if not fills.empty else 0
        else:
            slippage_mean = 0
        
        result = ReconciliationResult(
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            expected_return=expected_period_return,
            expected_sharpe=expected['sharpe'],
            expected_drawdown=expected['max_drawdown'],
            expected_turnover=int(expected_turnover),
            actual_return=actual['return'],
            actual_sharpe=actual['sharpe'],
            actual_drawdown=actual['drawdown'],
            actual_turnover=actual['turnover'],
            actual_slippage_mean=slippage_mean,
            return_divergence=return_div,
            sharpe_divergence=sharpe_div,
            drawdown_divergence=dd_div,
            turnover_divergence=turnover_div,
            status=status,
            warnings=warnings
        )
        
        # Save report
        self._save_report(result)
        
        return result
    
    def _save_report(self, result: ReconciliationResult):
        """Save reconciliation report."""
        # Generate filename with week number
        end_date = datetime.fromisoformat(result.period_end.replace('Z', '+00:00'))
        week_str = end_date.strftime("%Y-W%W")
        
        report_path = self.reports_dir / f"weekly_reconciliation_{week_str}.md"
        
        content = f"""# Weekly Reconciliation Report

**Period**: {result.period_start[:10]} to {result.period_end[:10]}
**Status**: **{result.status}**

## Summary

| Metric | Expected | Actual | Divergence |
|--------|----------|--------|------------|
| Return | {result.expected_return:.2f}% | {result.actual_return:.2f}% | {result.return_divergence:+.2f}% |
| Sharpe | {result.expected_sharpe:.2f} | {result.actual_sharpe:.2f} | {result.sharpe_divergence:+.2f} |
| Max DD | {result.expected_drawdown:.2f}% | {result.actual_drawdown:.2f}% | {result.drawdown_divergence:+.2f}% |
| Turnover | {result.expected_turnover} | {result.actual_turnover} | {result.turnover_divergence:+.0f} |

## Execution Quality

- **Mean Slippage**: {result.actual_slippage_mean:.2f} bps

## Warnings

"""
        if result.warnings:
            for w in result.warnings:
                content += f"- ⚠️ {w}\n"
        else:
            content += "None\n"
        
        content += f"""
## Notes

Generated: {datetime.now(timezone.utc).isoformat()}
"""
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Reconciliation report saved: {report_path}")
        
        # Also save JSON version
        json_path = self.reports_dir / f"weekly_reconciliation_{week_str}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'period_start': result.period_start,
                'period_end': result.period_end,
                'status': result.status,
                'expected': {
                    'return': result.expected_return,
                    'sharpe': result.expected_sharpe,
                    'drawdown': result.expected_drawdown,
                    'turnover': result.expected_turnover
                },
                'actual': {
                    'return': result.actual_return,
                    'sharpe': result.actual_sharpe,
                    'drawdown': result.actual_drawdown,
                    'turnover': result.actual_turnover,
                    'slippage_mean': result.actual_slippage_mean
                },
                'divergence': {
                    'return': result.return_divergence,
                    'sharpe': result.sharpe_divergence,
                    'drawdown': result.drawdown_divergence,
                    'turnover': result.turnover_divergence
                },
                'warnings': result.warnings
            }, f, indent=2)


if __name__ == "__main__":
    # Run reconciliation
    recon = WeeklyReconciliation()
    result = recon.reconcile()
    
    print(f"Status: {result.status}")
    print(f"Return: {result.actual_return:.2f}% (expected {result.expected_return:.2f}%)")
    print(f"Warnings: {result.warnings}")
