"""
Portfolio Risk Controls

Defines portfolio-level risk limits:
- Max concurrent exposure caps
- Correlation-aware position sizing
- Portfolio-level drawdown limits
- Portfolio kill-switch logic

Portfolio must fail SAFER than individual legs.

NO NEW ALPHA LOGIC. OBSERVE ONLY.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskLimits:
    """Portfolio-level risk limits."""
    max_total_exposure_pct: float  # Max % of capital exposed
    max_single_asset_pct: float  # Max % in any single asset
    max_correlated_exposure_pct: float  # Max in correlated assets
    portfolio_max_drawdown_pct: float  # Kill switch trigger
    max_assets: int  # Max number of concurrent positions


@dataclass
class RiskCheckResult:
    """Result of a risk limit check."""
    passed: bool
    limit_name: str
    current_value: float
    limit_value: float
    action_required: str  # "none", "reduce", "close"


@dataclass
class PortfolioRiskState:
    """Current portfolio risk state."""
    total_exposure_pct: float
    largest_position_pct: float
    correlated_exposure_pct: float
    current_drawdown_pct: float
    active_positions: int
    all_checks_passed: bool
    failed_checks: List[RiskCheckResult]


class PortfolioRiskControls:
    """
    Enforces portfolio-level risk limits.
    
    Usage:
        controls = PortfolioRiskControls()
        state = controls.check_all(positions, prices, equity_curve)
        if not state.all_checks_passed:
            # Take action
    """
    
    def __init__(
        self,
        limits: Optional[PortfolioRiskLimits] = None
    ):
        self.limits = limits or PortfolioRiskLimits(
            max_total_exposure_pct=150.0,  # 150% max (modest leverage)
            max_single_asset_pct=50.0,  # Max 50% in one asset
            max_correlated_exposure_pct=80.0,  # Max 80% in correlated assets
            portfolio_max_drawdown_pct=10.0,  # Kill switch at 10%
            max_assets=4  # Max 4 positions
        )
        
        logger.info(f"PortfolioRiskControls initialized")
        logger.info(f"  Max total exposure: {self.limits.max_total_exposure_pct}%")
        logger.info(f"  Portfolio kill switch: {self.limits.portfolio_max_drawdown_pct}% DD")
    
    def calculate_correlations(
        self,
        returns: Dict[str, pd.Series],
        lookback: int = 60
    ) -> pd.DataFrame:
        """Calculate correlation matrix from returns."""
        if not returns:
            return pd.DataFrame()
        
        # Align and combine
        df = pd.DataFrame(returns)
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        
        return df.corr()
    
    def check_total_exposure(
        self,
        positions: Dict[str, float],  # symbol: $ value
        total_capital: float
    ) -> RiskCheckResult:
        """Check total portfolio exposure."""
        total_exposure = sum(abs(v) for v in positions.values())
        exposure_pct = (total_exposure / total_capital * 100) if total_capital > 0 else 0
        
        passed = exposure_pct <= self.limits.max_total_exposure_pct
        
        return RiskCheckResult(
            passed=passed,
            limit_name="total_exposure",
            current_value=exposure_pct,
            limit_value=self.limits.max_total_exposure_pct,
            action_required="reduce" if not passed else "none"
        )
    
    def check_single_asset(
        self,
        positions: Dict[str, float],
        total_capital: float
    ) -> RiskCheckResult:
        """Check concentration in single asset."""
        if not positions or total_capital <= 0:
            return RiskCheckResult(
                passed=True,
                limit_name="single_asset",
                current_value=0,
                limit_value=self.limits.max_single_asset_pct,
                action_required="none"
            )
        
        largest = max(abs(v) for v in positions.values())
        largest_pct = (largest / total_capital * 100)
        
        passed = largest_pct <= self.limits.max_single_asset_pct
        
        return RiskCheckResult(
            passed=passed,
            limit_name="single_asset",
            current_value=largest_pct,
            limit_value=self.limits.max_single_asset_pct,
            action_required="reduce" if not passed else "none"
        )
    
    def check_correlated_exposure(
        self,
        positions: Dict[str, float],
        correlations: pd.DataFrame,
        total_capital: float,
        correlation_threshold: float = 0.7
    ) -> RiskCheckResult:
        """Check exposure in highly correlated assets."""
        if not positions or correlations.empty or total_capital <= 0:
            return RiskCheckResult(
                passed=True,
                limit_name="correlated_exposure",
                current_value=0,
                limit_value=self.limits.max_correlated_exposure_pct,
                action_required="none"
            )
        
        # Find correlated pairs
        max_correlated = 0
        for asset1 in positions:
            if asset1 not in correlations.index:
                continue
            correlated_exposure = abs(positions[asset1])
            for asset2 in positions:
                if asset1 == asset2 or asset2 not in correlations.columns:
                    continue
                if correlations.loc[asset1, asset2] > correlation_threshold:
                    correlated_exposure += abs(positions[asset2])
            max_correlated = max(max_correlated, correlated_exposure)
        
        correlated_pct = (max_correlated / total_capital * 100)
        passed = correlated_pct <= self.limits.max_correlated_exposure_pct
        
        return RiskCheckResult(
            passed=passed,
            limit_name="correlated_exposure",
            current_value=correlated_pct,
            limit_value=self.limits.max_correlated_exposure_pct,
            action_required="reduce" if not passed else "none"
        )
    
    def check_drawdown(
        self,
        equity_curve: pd.Series
    ) -> RiskCheckResult:
        """Check portfolio drawdown."""
        if equity_curve.empty:
            return RiskCheckResult(
                passed=True,
                limit_name="portfolio_drawdown",
                current_value=0,
                limit_value=self.limits.portfolio_max_drawdown_pct,
                action_required="none"
            )
        
        peak = equity_curve.cummax()
        drawdown = ((equity_curve - peak) / peak).iloc[-1] * 100
        
        passed = abs(drawdown) <= self.limits.portfolio_max_drawdown_pct
        
        return RiskCheckResult(
            passed=passed,
            limit_name="portfolio_drawdown",
            current_value=abs(drawdown),
            limit_value=self.limits.portfolio_max_drawdown_pct,
            action_required="close" if not passed else "none"
        )
    
    def check_position_count(
        self,
        positions: Dict[str, float]
    ) -> RiskCheckResult:
        """Check number of active positions."""
        active = sum(1 for v in positions.values() if abs(v) > 0)
        passed = active <= self.limits.max_assets
        
        return RiskCheckResult(
            passed=passed,
            limit_name="position_count",
            current_value=active,
            limit_value=self.limits.max_assets,
            action_required="reduce" if not passed else "none"
        )
    
    def check_all(
        self,
        positions: Dict[str, float],
        total_capital: float,
        equity_curve: pd.Series,
        correlations: Optional[pd.DataFrame] = None
    ) -> PortfolioRiskState:
        """Run all risk checks."""
        if correlations is None:
            correlations = pd.DataFrame()
        
        checks = [
            self.check_total_exposure(positions, total_capital),
            self.check_single_asset(positions, total_capital),
            self.check_correlated_exposure(positions, correlations, total_capital),
            self.check_drawdown(equity_curve),
            self.check_position_count(positions)
        ]
        
        failed = [c for c in checks if not c.passed]
        
        # Calculate state metrics
        total_exposure = sum(abs(v) for v in positions.values())
        largest = max(abs(v) for v in positions.values()) if positions else 0
        
        dd_check = self.check_drawdown(equity_curve)
        
        return PortfolioRiskState(
            total_exposure_pct=(total_exposure / total_capital * 100) if total_capital > 0 else 0,
            largest_position_pct=(largest / total_capital * 100) if total_capital > 0 else 0,
            correlated_exposure_pct=0,  # Would need correlation data
            current_drawdown_pct=dd_check.current_value,
            active_positions=sum(1 for v in positions.values() if abs(v) > 0),
            all_checks_passed=len(failed) == 0,
            failed_checks=failed
        )


if __name__ == "__main__":
    controls = PortfolioRiskControls()
    
    # Test positions
    positions = {
        "SPY": 50000,
        "GLD": 30000,
        "TLT": 20000
    }
    
    equity = pd.Series([100000, 98000, 95000, 96000], 
                       index=pd.date_range("2025-01-01", periods=4))
    
    state = controls.check_all(positions, 100000, equity)
    
    print(f"All checks passed: {state.all_checks_passed}")
    print(f"Total exposure: {state.total_exposure_pct:.1f}%")
    print(f"Drawdown: {state.current_drawdown_pct:.1f}%")
    if state.failed_checks:
        for check in state.failed_checks:
            print(f"  FAILED: {check.limit_name} ({check.current_value:.1f} > {check.limit_value:.1f})")
