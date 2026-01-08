"""
Permutation Tests

Block bootstrap tests preserving volatility clustering.
Returns randomized but regimes preserved.

The real strategy must outperform ≥95% of null variants on:
- Drawdown control
- Crash exposure

NOT total return.

DO NOT OPTIMIZE. ATTEMPT TO DESTROY.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PermutationResult:
    """Result of permutation testing."""
    n_permutations: int
    real_max_dd: float
    real_crash_exposure: float
    
    # Percentile ranks (higher = better, we beat more nulls)
    dd_percentile: float  # % of perms with worse DD
    crash_percentile: float  # % of perms with worse crash exposure
    
    # Distributions
    null_dd_distribution: List[float]
    null_crash_distribution: List[float]
    
    # Pass/Fail
    passes_dd_threshold: bool  # ≥95%
    passes_crash_threshold: bool  # ≥95%
    overall_pass: bool


class PermutationTester:
    """
    Performs permutation and block bootstrap tests.
    
    Tests whether the strategy's edge survives when:
    - Returns are randomized
    - But regimes (volatility clustering) are preserved
    """
    
    def __init__(
        self,
        n_permutations: int = 1000,
        block_size: int = 20,  # ~1 month of trading days
        pass_threshold: float = 0.95,
        seed: int = 42
    ):
        self.n_permutations = n_permutations
        self.block_size = block_size
        self.pass_threshold = pass_threshold
        self.seed = seed
        
        logger.info(f"PermutationTester initialized: {n_permutations} permutations")
    
    def block_bootstrap_returns(
        self,
        returns: pd.Series,
        block_size: Optional[int] = None
    ) -> pd.Series:
        """
        Generate block-bootstrapped returns preserving autocorrelation.
        """
        block_size = block_size or self.block_size
        n = len(returns)
        
        # Number of blocks needed
        n_blocks = int(np.ceil(n / block_size))
        
        # Available block starts
        max_start = n - block_size
        if max_start < 1:
            max_start = 1
        
        # Sample blocks
        block_starts = np.random.randint(0, max_start, size=n_blocks)
        
        # Build new series
        new_returns = []
        for start in block_starts:
            block = returns.iloc[start:start + block_size].values
            new_returns.extend(block)
        
        # Trim to original length
        new_returns = new_returns[:n]
        
        return pd.Series(new_returns, index=returns.index)
    
    def shuffle_preserving_regime(
        self,
        returns: pd.Series,
        regime_labels: pd.Series
    ) -> pd.Series:
        """
        Shuffle returns within each regime, preserving regime structure.
        """
        shuffled = returns.copy()
        
        for regime in regime_labels.unique():
            mask = regime_labels == regime
            regime_returns = returns[mask].values.copy()
            np.random.shuffle(regime_returns)
            shuffled.loc[mask] = regime_returns
        
        return shuffled
    
    def calculate_strategy_metrics(
        self,
        returns: pd.Series,
        positions: pd.Series,
        crash_threshold: float = -0.02
    ) -> Tuple[float, float]:
        """Calculate max drawdown and crash exposure."""
        strat_returns = returns * positions
        
        # Max drawdown
        equity = (1 + strat_returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        # Crash exposure
        crashes = returns < crash_threshold
        if crashes.sum() > 0:
            crash_exposure = positions[crashes].mean()
        else:
            crash_exposure = 0.0
        
        return float(max_dd), float(crash_exposure)
    
    def run_permutation_test(
        self,
        returns: pd.Series,
        positions: pd.Series,
        regime_labels: Optional[pd.Series] = None,
        method: str = "block_bootstrap"  # or "regime_shuffle"
    ) -> PermutationResult:
        """
        Run full permutation test.
        
        Args:
            returns: Market returns
            positions: Strategy positions (1 = long, 0 = flat)
            regime_labels: Optional regime labels for regime-preserving shuffle
            method: "block_bootstrap" or "regime_shuffle"
        """
        np.random.seed(self.seed)
        
        # Calculate real strategy metrics
        real_dd, real_crash = self.calculate_strategy_metrics(returns, positions)
        
        # Run permutations
        null_dds = []
        null_crashes = []
        
        for i in range(self.n_permutations):
            if method == "block_bootstrap":
                perm_returns = self.block_bootstrap_returns(returns)
            elif method == "regime_shuffle" and regime_labels is not None:
                perm_returns = self.shuffle_preserving_regime(returns, regime_labels)
            else:
                # Simple shuffle
                perm_returns = returns.sample(frac=1, replace=False)
                perm_returns.index = returns.index
            
            # Calculate metrics with same positions
            null_dd, null_crash = self.calculate_strategy_metrics(perm_returns, positions)
            null_dds.append(null_dd)
            null_crashes.append(null_crash)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Permutation {i+1}/{self.n_permutations}")
        
        # Calculate percentiles
        # For DD: real_dd is negative, more negative = worse
        # We want to know what % of nulls have WORSE (more negative) DD
        dd_percentile = np.mean([nd < real_dd for nd in null_dds])
        
        # For crash exposure: lower is better
        crash_percentile = np.mean([nc > real_crash for nc in null_crashes])
        
        passes_dd = dd_percentile >= self.pass_threshold
        passes_crash = crash_percentile >= self.pass_threshold
        
        return PermutationResult(
            n_permutations=self.n_permutations,
            real_max_dd=real_dd * 100,
            real_crash_exposure=real_crash * 100,
            dd_percentile=dd_percentile * 100,
            crash_percentile=crash_percentile * 100,
            null_dd_distribution=[d * 100 for d in null_dds],
            null_crash_distribution=[c * 100 for c in null_crashes],
            passes_dd_threshold=passes_dd,
            passes_crash_threshold=passes_crash,
            overall_pass=passes_dd and passes_crash
        )
    
    def generate_report(self, result: PermutationResult) -> str:
        """Generate markdown report."""
        pass_emoji = "✅" if result.overall_pass else "❌"
        
        content = f"""# Permutation Test Results

**N Permutations**: {result.n_permutations}
**Overall**: {pass_emoji} {"PASS" if result.overall_pass else "FAIL"}

---

## Real Strategy Metrics

| Metric | Value |
|--------|-------|
| Max Drawdown | {result.real_max_dd:.2f}% |
| Crash Exposure | {result.real_crash_exposure:.1f}% |

---

## Percentile Ranks (vs Null Distribution)

| Metric | Percentile | Pass (≥95%) |
|--------|------------|-------------|
| Drawdown Control | {result.dd_percentile:.1f}% | {"✅" if result.passes_dd_threshold else "❌"} |
| Crash Avoidance | {result.crash_percentile:.1f}% | {"✅" if result.passes_crash_threshold else "❌"} |

---

## Interpretation

"""
        if result.overall_pass:
            content += """The strategy's edge is **statistically significant**.

The drawdown control and crash avoidance beat ≥95% of random permutations,
meaning the edge is unlikely to be random luck.
"""
        else:
            content += """**WARNING**: The strategy's edge may be random.

The strategy did NOT beat ≥95% of random permutations on key metrics.
This is a potential falsification of the edge.

**Recommended Action**: Investigate before deploying capital.
"""
        
        content += f"""
---

## Null Distribution Summary

| Metric | Mean | Std | 5th Pct | 95th Pct |
|--------|------|-----|---------|----------|
| Null DD | {np.mean(result.null_dd_distribution):.2f}% | {np.std(result.null_dd_distribution):.2f}% | {np.percentile(result.null_dd_distribution, 5):.2f}% | {np.percentile(result.null_dd_distribution, 95):.2f}% |
| Null Crash | {np.mean(result.null_crash_distribution):.1f}% | {np.std(result.null_crash_distribution):.1f}% | {np.percentile(result.null_crash_distribution, 5):.1f}% | {np.percentile(result.null_crash_distribution, 95):.1f}% |
"""
        
        return content


if __name__ == "__main__":
    # Quick test
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    
    # Simulate returns with crashes
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    returns.iloc[50:55] = -0.03
    returns.iloc[200:210] = -0.025
    
    # Simulate positions (flat during crashes)
    positions = pd.Series(1, index=dates)
    positions.iloc[48:60] = 0
    positions.iloc[198:215] = 0
    
    tester = PermutationTester(n_permutations=100)  # Quick test
    result = tester.run_permutation_test(returns, positions)
    
    print(f"\nOverall Pass: {result.overall_pass}")
    print(f"DD Percentile: {result.dd_percentile:.1f}%")
    print(f"Crash Percentile: {result.crash_percentile:.1f}%")
