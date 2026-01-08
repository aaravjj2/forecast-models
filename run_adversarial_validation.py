#!/usr/bin/env python3
"""
Adversarial Validation Runner

Executes full adversarial validation suite:
1. Null Strategy Comparison
2. Permutation Tests (1000+)
3. Synthetic Path Survival
4. Portfolio Stress

The strategy must pass ALL tests or be considered FALSIFIED.

DO NOT OPTIMIZE. ATTEMPT TO DESTROY.
"""

import sys
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.price_fetcher import PriceFetcher
from features.feature_builder import FeatureBuilder
from features.regime_labeler import RegimeLabeler
from validation.null_strategies import NullStrategies, NullResult
from validation.permutation_tests import PermutationTester, PermutationResult
from validation.synthetic_paths import SyntheticPathGenerator
from validation.portfolio_stress import PortfolioStressTester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialValidator:
    """
    Runs complete adversarial validation suite.
    """
    
    def __init__(
        self,
        symbols: list = ["SPY", "GLD", "TLT", "QQQ"],
        lookback_years: int = 5,
        permutation_count: int = 1000,
        reports_dir: Optional[Path] = None
    ):
        self.symbols = symbols
        self.lookback_years = lookback_years
        self.permutation_count = permutation_count
        
        project_root = Path(__file__).parent
        self.reports_dir = reports_dir or project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.price_fetcher = PriceFetcher()
        self.feature_builder = FeatureBuilder()
        self.regime_labeler = RegimeLabeler()
        
        # Results storage
        self.results = {
            "null_comparison": None,
            "permutation_test": None,
            "synthetic_survival": None,
            "portfolio_stress": None,
            "overall_pass": False
        }
        
        logger.info(f"AdversarialValidator initialized for {symbols}")
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        logger.info("Fetching historical data...")
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=self.lookback_years * 365)).strftime("%Y-%m-%d")
        
        data = {}
        for symbol in self.symbols:
            prices = self.price_fetcher.fetch(symbol, start_date=start_date, end_date=end_date)
            data[symbol] = prices
            logger.info(f"  {symbol}: {len(prices)} days")
        
        return data
    
    def generate_regime_positions(
        self,
        prices: pd.DataFrame
    ) -> tuple:
        """Generate regime-based positions."""
        # Handle both 'Close' and 'close' column names
        close_col = 'Close' if 'Close' in prices.columns else 'close'
        returns = prices[close_col].pct_change().dropna()
        
        # Simple volatility regime (for demonstration)
        rolling_vol = returns.rolling(20).std()
        vol_threshold = rolling_vol.quantile(0.7)
        
        # Position: Long when vol below threshold
        positions = (rolling_vol < vol_threshold).astype(int)
        positions = positions.fillna(1)
        
        # Regime labels
        regimes = pd.Series("NEUTRAL", index=returns.index)
        regimes[rolling_vol > vol_threshold] = "HOSTILE"
        
        return returns, positions, regimes
    
    def run_null_comparison(
        self,
        returns: pd.Series,
        positions: pd.Series
    ) -> Dict:
        """Phase 10.1: Compare to null strategies."""
        logger.info("=" * 60)
        logger.info("PHASE 10.1: NULL STRATEGY COMPARISON")
        logger.info("=" * 60)
        
        nulls = NullStrategies()
        
        # Calculate real strategy result
        strat_returns = returns * positions
        equity = (1 + strat_returns).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        
        real_result = NullResult(
            name="VoltGated",
            total_return=(equity.iloc[-1] - 1) * 100,
            max_drawdown=float(dd.min()) * 100,
            sharpe=(strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() > 0 else 0,
            crash_exposure=0,
            turnover=int((positions.diff().abs() > 0).sum()),
            equity_curve=equity
        )
        
        # Run nulls
        null_results = nulls.run_all_nulls(returns, returns, positions)
        comparison = nulls.compare_to_real(real_result, null_results)
        
        # Report
        logger.info(f"\nReal Strategy: Return={real_result.total_return:.1f}%, DD={real_result.max_drawdown:.1f}%")
        
        wins_dd = 0
        for name, comp in comparison.items():
            logger.info(f"  vs {name}: DD improvement: {comp['dd_improvement']:.2f}%")
            if comp['real_wins_dd']:
                wins_dd += 1
        
        win_rate = wins_dd / len(comparison)
        passed = win_rate >= 0.8  # Beat 80% of nulls on DD
        
        logger.info(f"\nNull DD Win Rate: {win_rate:.1%} ({'PASS' if passed else 'FAIL'})")
        
        return {
            "real": real_result,
            "nulls": null_results,
            "comparison": comparison,
            "passed": passed
        }
    
    def run_permutation_test(
        self,
        returns: pd.Series,
        positions: pd.Series,
        regimes: pd.Series
    ) -> PermutationResult:
        """Phase 10.2: Permutation testing."""
        logger.info("=" * 60)
        logger.info(f"PHASE 10.2: PERMUTATION TEST ({self.permutation_count} permutations)")
        logger.info("=" * 60)
        
        tester = PermutationTester(n_permutations=self.permutation_count)
        result = tester.run_permutation_test(returns, positions, regimes, method="block_bootstrap")
        
        logger.info(f"\nReal DD: {result.real_max_dd:.2f}%")
        logger.info(f"DD Percentile: {result.dd_percentile:.1f}% {'PASS' if result.passes_dd_threshold else 'FAIL'}")
        logger.info(f"Crash Percentile: {result.crash_percentile:.1f}% {'PASS' if result.passes_crash_threshold else 'FAIL'}")
        logger.info(f"Overall: {'PASS' if result.overall_pass else 'FAIL'}")
        
        return result
    
    def run_synthetic_survival(
        self,
        returns: pd.Series,
        positions: pd.Series,
        regimes: pd.Series
    ) -> Dict:
        """Phase 11: Synthetic path survival."""
        logger.info("=" * 60)
        logger.info("PHASE 11: SYNTHETIC PATH SURVIVAL")
        logger.info("=" * 60)
        
        generator = SyntheticPathGenerator()
        paths = generator.generate_all_hostile_paths(returns, regimes, n_paths_per_type=10)
        
        catastrophic_threshold = -0.25
        survived = 0
        worst_dd = 0
        worst_path = ""
        
        for path in paths:
            # Apply same position logic to synthetic returns
            strat_returns = path.returns * positions.reindex(path.returns.index).fillna(1)
            equity = (1 + strat_returns).cumprod()
            peak = equity.cummax()
            dd = (equity - peak) / peak
            max_dd = float(dd.min())
            
            if max_dd > catastrophic_threshold:
                survived += 1
            
            if max_dd < worst_dd:
                worst_dd = max_dd
                worst_path = path.name
        
        survival_rate = survived / len(paths)
        passed = survival_rate >= 0.95  # 95% survival
        
        logger.info(f"\nPaths tested: {len(paths)}")
        logger.info(f"Survived: {survived}/{len(paths)} ({survival_rate:.1%})")
        logger.info(f"Worst DD: {worst_dd*100:.1f}% ({worst_path})")
        logger.info(f"Overall: {'PASS' if passed else 'FAIL'}")
        
        return {
            "paths_tested": len(paths),
            "survived": survived,
            "survival_rate": survival_rate,
            "worst_dd": worst_dd * 100,
            "worst_path": worst_path,
            "passed": passed
        }
    
    def run_portfolio_stress(
        self,
        data: Dict[str, pd.DataFrame],
        positions_dict: Dict[str, pd.Series]
    ) -> Dict:
        """Phase 12: Portfolio stress testing."""
        logger.info("=" * 60)
        logger.info("PHASE 12: PORTFOLIO STRESS TESTING")
        logger.info("=" * 60)
        
        returns_dict = {}
        for symbol, prices in data.items():
            close_col = 'Close' if 'Close' in prices.columns else 'close'
            returns_dict[symbol] = prices[close_col].pct_change().dropna()
        
        # Align indices
        common_idx = returns_dict[self.symbols[0]].index
        for symbol in self.symbols[1:]:
            common_idx = common_idx.intersection(returns_dict[symbol].index)
        
        aligned_returns = {s: returns_dict[s].loc[common_idx] for s in self.symbols}
        aligned_positions = {s: positions_dict[s].reindex(common_idx).fillna(1) for s in self.symbols}
        
        tester = PortfolioStressTester()
        summary = tester.run_all_scenarios(aligned_returns, aligned_positions)
        
        logger.info(f"\nScenarios: {summary.scenarios_passed}/{summary.total_scenarios}")
        logger.info(f"Survival Rate: {summary.survival_rate:.1%}")
        logger.info(f"Worst DD: {summary.worst_dd:.1f}% ({summary.worst_scenario})")
        logger.info(f"Kill-Switch Accuracy: {summary.kill_switch_accuracy:.1%}")
        logger.info(f"Overall: {'PASS' if summary.overall_pass else 'FAIL'}")
        
        return {
            "summary": summary,
            "passed": summary.overall_pass
        }
    
    def generate_final_report(self) -> Path:
        """Generate comprehensive adversarial validation report."""
        report_path = self.reports_dir / f"adversarial_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        
        overall = all([
            self.results.get("null_comparison", {}).get("passed", False),
            self.results.get("permutation_test") and self.results["permutation_test"].overall_pass,
            self.results.get("synthetic_survival", {}).get("passed", False),
            self.results.get("portfolio_stress", {}).get("passed", False)
        ])
        
        self.results["overall_pass"] = overall
        
        verdict = "✅ PASS: Edge survives adversarial testing" if overall else "❌ FAIL: Edge FALSIFIED"
        
        content = f"""# Adversarial Validation Report

**Generated**: {datetime.now(timezone.utc).isoformat()}
**Verdict**: {verdict}

---

## Summary

| Phase | Test | Result |
|-------|------|--------|
| 10.1 | Null Strategy Comparison | {"✅" if self.results.get("null_comparison", {}).get("passed") else "❌"} |
| 10.2 | Permutation Test (1000+) | {"✅" if self.results.get("permutation_test") and self.results["permutation_test"].overall_pass else "❌"} |
| 11 | Synthetic Path Survival | {"✅" if self.results.get("synthetic_survival", {}).get("passed") else "❌"} |
| 12 | Portfolio Stress | {"✅" if self.results.get("portfolio_stress", {}).get("passed") else "❌"} |

---

## Phase 10.1: Null Strategy Comparison

"""
        if self.results.get("null_comparison"):
            nc = self.results["null_comparison"]
            content += f"Real Strategy DD: {nc['real'].max_drawdown:.2f}%\n\n"
            content += "| Null Strategy | DD Improvement | Win |\n|---------------|----------------|-----|\n"
            for name, comp in nc["comparison"].items():
                content += f"| {name} | {comp['dd_improvement']:+.2f}% | {'✅' if comp['real_wins_dd'] else '❌'} |\n"
        
        content += """
---

## Phase 10.2: Permutation Test

"""
        if self.results.get("permutation_test"):
            pt = self.results["permutation_test"]
            content += f"- Permutations: {pt.n_permutations}\n"
            content += f"- DD Percentile: {pt.dd_percentile:.1f}% (≥95% required)\n"
            content += f"- Crash Percentile: {pt.crash_percentile:.1f}% (≥95% required)\n"
        
        content += """
---

## Phase 11: Synthetic Path Survival

"""
        if self.results.get("synthetic_survival"):
            ss = self.results["synthetic_survival"]
            content += f"- Paths Tested: {ss['paths_tested']}\n"
            content += f"- Survival Rate: {ss['survival_rate']:.1%}\n"
            content += f"- Worst DD: {ss['worst_dd']:.1f}% ({ss['worst_path']})\n"
        
        content += """
---

## Phase 12: Portfolio Stress

"""
        if self.results.get("portfolio_stress"):
            ps = self.results["portfolio_stress"]["summary"]
            content += f"- Scenarios: {ps.scenarios_passed}/{ps.total_scenarios}\n"
            content += f"- Survival Rate: {ps.survival_rate:.1%}\n"
            content += f"- Worst DD: {ps.worst_dd:.1f}%\n"
        
        content += f"""
---

## Final Verdict

**{verdict}**

"""
        if overall:
            content += """The Volatility-Gated Long Exposure strategy has survived all adversarial tests.
The edge appears robust to:
- Random baselines
- Permutation shuffling
- Synthetic hostile regimes
- Multi-asset stress scenarios

**Recommendation**: Proceed to Phase 13 (Deployment Decision).
"""
        else:
            content += """**WARNING**: The strategy has FAILED one or more adversarial tests.

The edge may be:
- Random luck that survived backtesting
- Sensitive to regime changes not captured
- Unable to survive portfolio-level stress

**Recommendation**: DO NOT DEPLOY CAPITAL. Investigate failures.
"""
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved: {report_path}")
        return report_path
    
    def run_full_validation(self) -> bool:
        """Run complete adversarial validation suite."""
        logger.info("=" * 60)
        logger.info("ADVERSARIAL VALIDATION SUITE")
        logger.info("Attempting to DESTROY the strategy...")
        logger.info("=" * 60)
        
        # Fetch data
        data = self.fetch_data()
        
        # Generate positions for primary symbol
        primary = self.symbols[0]
        returns, positions, regimes = self.generate_regime_positions(data[primary])
        
        # Generate positions for all symbols
        positions_dict = {}
        for symbol in self.symbols:
            _, pos, _ = self.generate_regime_positions(data[symbol])
            positions_dict[symbol] = pos
        
        # Phase 10.1: Null comparison
        self.results["null_comparison"] = self.run_null_comparison(returns, positions)
        
        # Phase 10.2: Permutation tests
        self.results["permutation_test"] = self.run_permutation_test(returns, positions, regimes)
        
        # Phase 11: Synthetic survival
        self.results["synthetic_survival"] = self.run_synthetic_survival(returns, positions, regimes)
        
        # Phase 12: Portfolio stress
        self.results["portfolio_stress"] = self.run_portfolio_stress(data, positions_dict)
        
        # Generate report
        report_path = self.generate_final_report()
        
        logger.info("=" * 60)
        if self.results["overall_pass"]:
            logger.info("✅ VALIDATION PASSED: Edge survives adversarial testing")
        else:
            logger.info("❌ VALIDATION FAILED: Edge FALSIFIED")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 60)
        
        return self.results["overall_pass"]


def main():
    parser = argparse.ArgumentParser(description="Run adversarial validation suite")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "GLD", "TLT", "QQQ"])
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--lookback-years", type=int, default=5)
    
    args = parser.parse_args()
    
    validator = AdversarialValidator(
        symbols=args.symbols,
        permutation_count=args.permutations,
        lookback_years=args.lookback_years
    )
    
    passed = validator.run_full_validation()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
