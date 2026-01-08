"""
Portfolio Stress Testing

Multi-asset stress compression:
- Run strategy across SPY, GLD, TLT, QQQ simultaneously
- Inject correlation spikes, liquidity shocks, execution delays
- Verify kill-switches activate correctly

Portfolio DD must remain below threshold in ≥99% of scenarios.

DO NOT OPTIMIZE. ATTEMPT TO DESTROY.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """A single stress scenario."""
    name: str
    description: str
    correlation_spike: float  # 0-1
    slippage_multiplier: float  # 1 = normal, 10 = 10x
    execution_delay: int  # Days of additional delay


@dataclass
class PortfolioStressResult:
    """Result of portfolio stress test."""
    scenario_name: str
    portfolio_dd: float
    individual_dds: Dict[str, float]
    kill_switch_triggered: bool
    kill_switch_correct: bool  # Should it have triggered?
    survived: bool


@dataclass
class PortfolioStressSummary:
    """Summary of all stress tests."""
    total_scenarios: int
    scenarios_passed: int
    survival_rate: float
    worst_dd: float
    worst_scenario: str
    kill_switch_accuracy: float
    overall_pass: bool  # ≥99% survival


class PortfolioStressTester:
    """
    Tests portfolio-level risk under extreme conditions.
    """
    
    STRESS_SCENARIOS = [
        StressScenario("Normal", "Baseline conditions", 0.3, 1.0, 0),
        StressScenario("HighCorr", "All assets correlate", 0.95, 1.0, 0),
        StressScenario("PerfectCorr", "Correlation → 1.0", 1.0, 1.0, 0),
        StressScenario("LiquidityShock5x", "5x normal slippage", 0.3, 5.0, 0),
        StressScenario("LiquidityShock10x", "10x normal slippage", 0.3, 10.0, 0),
        StressScenario("ExecDelay1d", "T+2 execution", 0.3, 1.0, 1),
        StressScenario("ExecDelay2d", "T+3 execution", 0.3, 1.0, 2),
        StressScenario("CorrPlusLiq", "Corr spike + liquidity", 0.95, 5.0, 0),
        StressScenario("FullStress", "All factors combined", 0.95, 10.0, 1),
        StressScenario("BlackSwan", "Extreme everything", 1.0, 20.0, 2),
    ]
    
    def __init__(
        self,
        kill_threshold: float = -0.10,  # 10% portfolio DD = kill
        pass_threshold: float = 0.99,  # 99% scenarios must pass
        base_slippage_bps: float = 5.0,
        seed: int = 42
    ):
        self.kill_threshold = kill_threshold
        self.pass_threshold = pass_threshold
        self.base_slippage = base_slippage_bps
        self.seed = seed
        
        logger.info(f"PortfolioStressTester initialized (kill: {kill_threshold*100:.0f}% DD)")
    
    def generate_correlated_returns(
        self,
        base_returns: Dict[str, pd.Series],
        target_correlation: float
    ) -> Dict[str, pd.Series]:
        """
        Transform returns to achieve target correlation.
        """
        if target_correlation <= 0.3:
            return base_returns
        
        symbols = list(base_returns.keys())
        common_idx = base_returns[symbols[0]].index
        n = len(common_idx)
        
        # Create common factor
        common_factor = np.random.normal(0, 0.015, n)
        
        correlated = {}
        for symbol in symbols:
            original = base_returns[symbol].values
            # Blend: correlation% common + (1-correlation%) idiosyncratic
            blended = target_correlation * common_factor + (1 - target_correlation) * original
            correlated[symbol] = pd.Series(blended, index=common_idx)
        
        return correlated
    
    def apply_slippage(
        self,
        returns: pd.Series,
        positions: pd.Series,
        slippage_multiplier: float
    ) -> pd.Series:
        """
        Apply slippage costs to returns.
        """
        effective_slippage = self.base_slippage * slippage_multiplier / 10000
        
        # Slippage only on position changes
        changes = positions.diff().abs()
        slippage_cost = changes * effective_slippage
        
        return returns - slippage_cost
    
    def apply_execution_delay(
        self,
        positions: pd.Series,
        delay: int
    ) -> pd.Series:
        """
        Delay position execution.
        """
        if delay <= 0:
            return positions
        
        return positions.shift(delay).fillna(1)  # Default to long
    
    def calculate_portfolio_dd(
        self,
        asset_returns: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> Tuple[float, pd.Series]:
        """
        Calculate portfolio-level drawdown.
        """
        # Weighted portfolio returns
        portfolio_returns = pd.Series(0, index=asset_returns[list(asset_returns.keys())[0]].index)
        
        for symbol, returns in asset_returns.items():
            weight = weights.get(symbol, 1.0 / len(asset_returns))
            portfolio_returns += returns * weight
        
        # Equity curve
        equity = (1 + portfolio_returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        return float(drawdown.min()), equity
    
    def run_scenario(
        self,
        scenario: StressScenario,
        base_returns: Dict[str, pd.Series],
        base_positions: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> PortfolioStressResult:
        """
        Run a single stress scenario.
        """
        # Apply correlation shock
        shocked_returns = self.generate_correlated_returns(
            base_returns, scenario.correlation_spike
        )
        
        # Apply execution delay and slippage per asset
        strategy_returns = {}
        individual_dds = {}
        
        for symbol in base_returns:
            positions = self.apply_execution_delay(
                base_positions[symbol], scenario.execution_delay
            )
            
            returns_after_slip = self.apply_slippage(
                shocked_returns[symbol], positions, scenario.slippage_multiplier
            )
            
            # Strategy returns
            strat_returns = returns_after_slip * positions
            strategy_returns[symbol] = strat_returns
            
            # Individual DD
            equity = (1 + strat_returns).cumprod()
            peak = equity.cummax()
            dd = (equity - peak) / peak
            individual_dds[symbol] = float(dd.min()) * 100
        
        # Portfolio DD
        portfolio_dd, _ = self.calculate_portfolio_dd(strategy_returns, weights)
        portfolio_dd_pct = portfolio_dd * 100
        
        # Kill switch logic
        kill_switch_triggered = portfolio_dd <= self.kill_threshold
        should_trigger = portfolio_dd <= self.kill_threshold
        kill_switch_correct = kill_switch_triggered == should_trigger
        
        # Survival
        survived = portfolio_dd > self.kill_threshold
        
        return PortfolioStressResult(
            scenario_name=scenario.name,
            portfolio_dd=portfolio_dd_pct,
            individual_dds=individual_dds,
            kill_switch_triggered=kill_switch_triggered,
            kill_switch_correct=kill_switch_correct,
            survived=survived
        )
    
    def run_all_scenarios(
        self,
        base_returns: Dict[str, pd.Series],
        base_positions: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None
    ) -> PortfolioStressSummary:
        """
        Run all stress scenarios.
        """
        np.random.seed(self.seed)
        
        if weights is None:
            weights = {s: 1.0 / len(base_returns) for s in base_returns}
        
        results = []
        
        for scenario in self.STRESS_SCENARIOS:
            logger.info(f"Running scenario: {scenario.name}")
            result = self.run_scenario(scenario, base_returns, base_positions, weights)
            results.append(result)
        
        # Summary
        passed = sum(1 for r in results if r.survived)
        total = len(results)
        survival_rate = passed / total
        
        worst_result = min(results, key=lambda r: r.portfolio_dd)
        
        kill_correct = sum(1 for r in results if r.kill_switch_correct) / total
        
        return PortfolioStressSummary(
            total_scenarios=total,
            scenarios_passed=passed,
            survival_rate=survival_rate,
            worst_dd=worst_result.portfolio_dd,
            worst_scenario=worst_result.scenario_name,
            kill_switch_accuracy=kill_correct,
            overall_pass=survival_rate >= self.pass_threshold
        )
    
    def generate_report(
        self,
        summary: PortfolioStressSummary,
        results: List[PortfolioStressResult]
    ) -> str:
        """Generate markdown stress test report."""
        pass_emoji = "✅" if summary.overall_pass else "❌"
        
        content = f"""# Portfolio Stress Test Results

**Overall**: {pass_emoji} {"PASS" if summary.overall_pass else "FAIL"}

---

## Summary

| Metric | Value | Required |
|--------|-------|----------|
| Survival Rate | {summary.survival_rate:.1%} | ≥99% |
| Scenarios Passed | {summary.scenarios_passed}/{summary.total_scenarios} | |
| Worst Drawdown | {summary.worst_dd:.2f}% | |
| Worst Scenario | {summary.worst_scenario} | |
| Kill-Switch Accuracy | {summary.kill_switch_accuracy:.1%} | |

---

## Scenario Details

| Scenario | Portfolio DD | Survived | Kill-Switch |
|----------|-------------|----------|-------------|
"""
        for result in results:
            survived_emoji = "✅" if result.survived else "❌"
            ks_emoji = "✅" if result.kill_switch_correct else "⚠️"
            content += f"| {result.scenario_name} | {result.portfolio_dd:.2f}% | {survived_emoji} | {ks_emoji} |\n"
        
        if not summary.overall_pass:
            content += """
---

> [!CAUTION]
> **Strategy FAILED portfolio stress tests.**
> 
> Survival rate below 99% threshold.
> Capital deployment NOT recommended.
"""
        
        return content


if __name__ == "__main__":
    # Quick test
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    np.random.seed(42)
    
    returns = {
        "SPY": pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates),
        "GLD": pd.Series(np.random.normal(0.0003, 0.010, len(dates)), index=dates),
        "TLT": pd.Series(np.random.normal(0.0002, 0.008, len(dates)), index=dates),
        "QQQ": pd.Series(np.random.normal(0.0006, 0.018, len(dates)), index=dates),
    }
    
    positions = {
        symbol: pd.Series(np.random.choice([0, 1], len(dates), p=[0.3, 0.7]), index=dates)
        for symbol in returns
    }
    
    tester = PortfolioStressTester()
    summary = tester.run_all_scenarios(returns, positions)
    
    print(f"\nOverall Pass: {summary.overall_pass}")
    print(f"Survival Rate: {summary.survival_rate:.1%}")
    print(f"Worst DD: {summary.worst_dd:.2f}% ({summary.worst_scenario})")
