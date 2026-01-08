"""
Null Strategies

Baseline strategies to compare against the Volatility-Gated strategy.
If the real strategy cannot beat these on DRAWDOWN CONTROL, the edge is falsified.

Null Set:
1. Random Entry/Exit - Same turnover, random timing
2. Always-Long - 100% exposure, no regime gating
3. Delayed Signal - T+2, T+3 execution variants
4. Inverse Signal - Opposite of regime predictions

DO NOT OPTIMIZE. ATTEMPT TO DESTROY.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NullResult:
    """Result of running a null strategy."""
    name: str
    total_return: float
    max_drawdown: float
    sharpe: float
    crash_exposure: float  # % of time exposed during crashes
    turnover: int
    equity_curve: pd.Series


class NullStrategies:
    """
    Generates null strategy baselines for comparison.
    
    The real strategy must outperform these on DRAWDOWN CONTROL,
    not total return.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        logger.info("NullStrategies initialized")
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        positions: pd.Series,
        market_returns: pd.Series,
        crash_threshold: float = -0.02
    ) -> Tuple[float, float, float, float, int]:
        """Calculate strategy metrics."""
        # Strategy returns
        strat_returns = returns * positions
        
        # Total return
        total_return = (1 + strat_returns).prod() - 1
        
        # Equity curve
        equity = (1 + strat_returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        # Sharpe
        if strat_returns.std() > 0:
            sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Crash exposure
        crashes = market_returns < crash_threshold
        if crashes.sum() > 0:
            crash_exposure = positions[crashes].mean()
        else:
            crash_exposure = 0.0
        
        # Turnover
        turnover = (positions.diff().abs() > 0).sum()
        
        return float(total_return), float(max_dd), float(sharpe), float(crash_exposure), int(turnover)
    
    def random_strategy(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        real_turnover: int
    ) -> NullResult:
        """
        Random entry/exit with same turnover as real strategy.
        """
        n = len(returns)
        positions = pd.Series(1, index=returns.index)  # Start long
        
        # Random position changes at matching frequency
        change_prob = real_turnover / n
        changes = np.random.random(n) < change_prob
        
        current_pos = 1
        for i, change in enumerate(changes):
            if change:
                current_pos = 1 - current_pos  # Flip
            positions.iloc[i] = current_pos
        
        total_ret, max_dd, sharpe, crash_exp, turnover = self._calculate_metrics(
            returns, positions, market_returns
        )
        
        equity = (1 + returns * positions).cumprod()
        
        return NullResult(
            name="Random",
            total_return=total_ret * 100,
            max_drawdown=max_dd * 100,
            sharpe=sharpe,
            crash_exposure=crash_exp * 100,
            turnover=turnover,
            equity_curve=equity
        )
    
    def always_long_strategy(
        self,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> NullResult:
        """
        Always 100% long. No regime gating.
        """
        positions = pd.Series(1, index=returns.index)
        
        total_ret, max_dd, sharpe, crash_exp, turnover = self._calculate_metrics(
            returns, positions, market_returns
        )
        
        equity = (1 + returns * positions).cumprod()
        
        return NullResult(
            name="Always-Long",
            total_return=total_ret * 100,
            max_drawdown=max_dd * 100,
            sharpe=sharpe,
            crash_exposure=crash_exp * 100,
            turnover=0,
            equity_curve=equity
        )
    
    def delayed_signal_strategy(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        real_positions: pd.Series,
        delay: int = 2
    ) -> NullResult:
        """
        Real signal but delayed by T+delay days.
        """
        positions = real_positions.shift(delay).fillna(1)
        
        total_ret, max_dd, sharpe, crash_exp, turnover = self._calculate_metrics(
            returns, positions, market_returns
        )
        
        equity = (1 + returns * positions).cumprod()
        
        return NullResult(
            name=f"Delayed-T+{delay}",
            total_return=total_ret * 100,
            max_drawdown=max_dd * 100,
            sharpe=sharpe,
            crash_exposure=crash_exp * 100,
            turnover=turnover,
            equity_curve=equity
        )
    
    def inverse_signal_strategy(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        real_positions: pd.Series
    ) -> NullResult:
        """
        Opposite of real regime predictions.
        """
        positions = 1 - real_positions  # Flip: 1→0, 0→1
        
        total_ret, max_dd, sharpe, crash_exp, turnover = self._calculate_metrics(
            returns, positions, market_returns
        )
        
        equity = (1 + returns * positions).cumprod()
        
        return NullResult(
            name="Inverse",
            total_return=total_ret * 100,
            max_drawdown=max_dd * 100,
            sharpe=sharpe,
            crash_exposure=crash_exp * 100,
            turnover=turnover,
            equity_curve=equity
        )
    
    def run_all_nulls(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        real_positions: pd.Series
    ) -> Dict[str, NullResult]:
        """Run all null strategies."""
        real_turnover = (real_positions.diff().abs() > 0).sum()
        
        results = {}
        
        # Random (run multiple times for robustness)
        random_results = []
        for i in range(10):
            np.random.seed(self.seed + i)
            random_results.append(self.random_strategy(returns, market_returns, real_turnover))
        # Take median performance random
        median_idx = np.argsort([r.max_drawdown for r in random_results])[5]
        results["Random"] = random_results[median_idx]
        
        # Always-Long
        results["Always-Long"] = self.always_long_strategy(returns, market_returns)
        
        # Delayed signals
        results["Delayed-T+2"] = self.delayed_signal_strategy(returns, market_returns, real_positions, delay=2)
        results["Delayed-T+3"] = self.delayed_signal_strategy(returns, market_returns, real_positions, delay=3)
        
        # Inverse
        results["Inverse"] = self.inverse_signal_strategy(returns, market_returns, real_positions)
        
        return results
    
    def compare_to_real(
        self,
        real_result: NullResult,
        null_results: Dict[str, NullResult]
    ) -> Dict[str, Dict]:
        """Compare real strategy to null baselines."""
        comparison = {}
        
        for name, null in null_results.items():
            comparison[name] = {
                "dd_improvement": real_result.max_drawdown - null.max_drawdown,
                "crash_improvement": null.crash_exposure - real_result.crash_exposure,
                "return_delta": real_result.total_return - null.total_return,
                "real_wins_dd": real_result.max_drawdown > null.max_drawdown,  # Less negative = better
                "real_wins_crash": real_result.crash_exposure < null.crash_exposure
            }
        
        return comparison


if __name__ == "__main__":
    # Quick test with synthetic data
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    
    # Simulate market with some crashes
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    returns.iloc[50:55] = -0.03  # Crash 1
    returns.iloc[200:210] = -0.025  # Crash 2
    
    # Simulate real strategy positions (flat during crashes)
    positions = pd.Series(1, index=dates)
    positions.iloc[48:60] = 0  # Flat around crash 1
    positions.iloc[198:215] = 0  # Flat around crash 2
    
    nulls = NullStrategies()
    null_results = nulls.run_all_nulls(returns, returns, positions)
    
    print("\nNull Strategy Results:")
    print("-" * 60)
    for name, result in null_results.items():
        print(f"{name:15} | Return: {result.total_return:+.1f}% | DD: {result.max_drawdown:.1f}% | Crash Exp: {result.crash_exposure:.0f}%")
