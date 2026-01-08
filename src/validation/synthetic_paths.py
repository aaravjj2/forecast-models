"""
Synthetic Path Generator

Compresses decades of regimes into hours of computation.

Generates hostile market paths using:
- Volatility Regime Bootstrap
- Crisis Stitching (2008-style tails)
- Correlation Shock Overlays
- Whipsaw Volatility
- Prolonged False Risk-Off

The strategy must NEVER catastrophically fail across ALL synthetic histories.

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
class SyntheticPath:
    """A synthetic market path."""
    name: str
    returns: pd.Series
    regime_labels: pd.Series
    description: str


@dataclass
class SurvivalResult:
    """Result of running strategy on synthetic paths."""
    path_name: str
    max_drawdown: float
    final_return: float
    survived: bool  # No catastrophic DD
    false_exits: int  # Exited when should have stayed
    missed_entries: int  # Stayed out when should have entered


class SyntheticPathGenerator:
    """
    Generates hostile synthetic market paths.
    
    The goal is to break the strategy.
    """
    
    def __init__(
        self,
        catastrophic_dd_threshold: float = -0.25,  # 25% = catastrophic
        seed: int = 42
    ):
        self.catastrophic_threshold = catastrophic_dd_threshold
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(f"SyntheticPathGenerator initialized (catastrophic DD: {catastrophic_dd_threshold*100:.0f}%)")
    
    def regime_bootstrap(
        self,
        historical_returns: pd.Series,
        historical_regimes: pd.Series,
        n_days: int = 500
    ) -> SyntheticPath:
        """
        Bootstrap by sampling entire regime blocks.
        Preserves volatility clustering.
        """
        # Find regime runs
        runs = []
        current_regime = historical_regimes.iloc[0]
        current_start = 0
        
        for i in range(1, len(historical_regimes)):
            if historical_regimes.iloc[i] != current_regime:
                runs.append((current_regime, current_start, i))
                current_regime = historical_regimes.iloc[i]
                current_start = i
        runs.append((current_regime, current_start, len(historical_regimes)))
        
        # Sample runs until we have enough days
        synthetic_returns = []
        synthetic_regimes = []
        
        while len(synthetic_returns) < n_days:
            run = runs[np.random.randint(0, len(runs))]
            regime, start, end = run
            
            block_returns = historical_returns.iloc[start:end].values
            block_regimes = [regime] * len(block_returns)
            
            synthetic_returns.extend(block_returns)
            synthetic_regimes.extend(block_regimes)
        
        # Trim to exact length
        synthetic_returns = synthetic_returns[:n_days]
        synthetic_regimes = synthetic_regimes[:n_days]
        
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        
        return SyntheticPath(
            name="RegimeBootstrap",
            returns=pd.Series(synthetic_returns, index=dates),
            regime_labels=pd.Series(synthetic_regimes, index=dates),
            description="Block-sampled regime periods preserving clustering"
        )
    
    def crisis_stitch(
        self,
        normal_returns: pd.Series,
        crisis_returns: pd.Series,
        n_crises: int = 3,
        crisis_length: int = 20
    ) -> SyntheticPath:
        """
        Stitch crisis periods into normal market.
        Simulates 2008-style sudden drawdowns.
        """
        synthetic = normal_returns.copy()
        n = len(synthetic)
        
        # Insert crises at random points
        for _ in range(n_crises):
            insert_point = np.random.randint(crisis_length, n - crisis_length)
            
            # Sample crisis returns (scale to be severe)
            crisis_block = crisis_returns.sample(min(crisis_length, len(crisis_returns)), replace=True).values
            crisis_block = crisis_block * 1.5  # Amplify
            
            synthetic.iloc[insert_point:insert_point + len(crisis_block)] = crisis_block
        
        # Create regime labels (HOSTILE during crisis)
        regimes = pd.Series("NEUTRAL", index=synthetic.index)
        rolling_vol = synthetic.rolling(window=5).std()
        regimes[rolling_vol > rolling_vol.quantile(0.8)] = "HOSTILE"
        
        return SyntheticPath(
            name="CrisisStitch",
            returns=synthetic,
            regime_labels=regimes,
            description=f"{n_crises} crisis periods stitched into normal market"
        )
    
    def back_to_back_crisis(
        self,
        crisis_returns: pd.Series,
        n_days: int = 100
    ) -> SyntheticPath:
        """
        Pure crisis scenario - no recovery periods.
        """
        # Sample crisis returns repeatedly
        synthetic_returns = []
        
        while len(synthetic_returns) < n_days:
            block = crisis_returns.sample(min(20, len(crisis_returns)), replace=True).values
            synthetic_returns.extend(block)
        
        synthetic_returns = synthetic_returns[:n_days]
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        
        return SyntheticPath(
            name="BackToBackCrisis",
            returns=pd.Series(synthetic_returns, index=dates),
            regime_labels=pd.Series("HOSTILE", index=dates),
            description="Continuous crisis with no recovery - maximum stress"
        )
    
    def whipsaw(
        self,
        returns: pd.Series,
        flip_frequency: int = 3  # Days between regime flips
    ) -> SyntheticPath:
        """
        Rapid regime flipping - maximum false signals.
        """
        n = len(returns)
        regimes = []
        current = "NEUTRAL"
        
        for i in range(n):
            if i % flip_frequency == 0:
                current = "HOSTILE" if current == "NEUTRAL" else "NEUTRAL"
            regimes.append(current)
        
        # Amplify volatility during regime changes
        synthetic = returns.copy()
        for i in range(len(regimes) - 1):
            if regimes[i] != regimes[i+1]:
                synthetic.iloc[i] = synthetic.iloc[i] * np.random.uniform(1.5, 2.5)
        
        return SyntheticPath(
            name="Whipsaw",
            returns=synthetic,
            regime_labels=pd.Series(regimes, index=returns.index),
            description=f"Regime flip every {flip_frequency} days - false signal stress"
        )
    
    def prolonged_false_risk_off(
        self,
        returns: pd.Series,
        false_period: int = 60
    ) -> SyntheticPath:
        """
        Market looks risky but actually rallies.
        Maximum opportunity cost scenario.
        """
        n = len(returns)
        
        # Create upward trending returns
        synthetic = pd.Series(np.random.normal(0.002, 0.01, n), index=returns.index)
        
        # But label as HOSTILE
        regimes = pd.Series("HOSTILE", index=returns.index)
        
        return SyntheticPath(
            name="FalseRiskOff",
            returns=synthetic,
            regime_labels=regimes,
            description="Rally disguised as crisis - maximum missed opportunity"
        )
    
    def correlation_shock(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> Dict[str, SyntheticPath]:
        """
        Force all assets to correlate → 1.0.
        Tests portfolio diversification failure.
        """
        # Get common index
        common_idx = returns_dict[list(returns_dict.keys())[0]].index
        
        # Generate single return series
        base_returns = pd.Series(np.random.normal(0, 0.02, len(common_idx)), index=common_idx)
        
        # All assets follow same pattern (correlation → 1)
        shocked_paths = {}
        for symbol in returns_dict:
            noise = np.random.normal(0, 0.002, len(common_idx))
            shocked_paths[symbol] = SyntheticPath(
                name=f"CorrShock_{symbol}",
                returns=base_returns + pd.Series(noise, index=common_idx),
                regime_labels=pd.Series("NEUTRAL", index=common_idx),
                description="All assets perfectly correlated"
            )
        
        return shocked_paths
    
    def generate_all_hostile_paths(
        self,
        historical_returns: pd.Series,
        historical_regimes: pd.Series,
        n_paths_per_type: int = 10
    ) -> List[SyntheticPath]:
        """
        Generate full suite of hostile synthetic paths.
        """
        paths = []
        
        # Identify crisis returns (bottom 20% of vol)
        rolling_vol = historical_returns.rolling(20).std()
        crisis_mask = rolling_vol > rolling_vol.quantile(0.8)
        crisis_returns = historical_returns[crisis_mask]
        
        if len(crisis_returns) < 20:
            crisis_returns = historical_returns[historical_returns < historical_returns.quantile(0.1)]
        
        logger.info("Generating hostile synthetic paths...")
        
        # Regime bootstrap paths
        for i in range(n_paths_per_type):
            np.random.seed(self.seed + i)
            paths.append(self.regime_bootstrap(historical_returns, historical_regimes))
        
        # Crisis stitch paths
        for i in range(n_paths_per_type):
            np.random.seed(self.seed + 100 + i)
            paths.append(self.crisis_stitch(historical_returns, crisis_returns, n_crises=3))
        
        # Back to back crisis (few of these - extreme)
        for i in range(3):
            np.random.seed(self.seed + 200 + i)
            paths.append(self.back_to_back_crisis(crisis_returns))
        
        # Whipsaw paths
        for i in range(n_paths_per_type):
            np.random.seed(self.seed + 300 + i)
            paths.append(self.whipsaw(historical_returns, flip_frequency=3 + i))
        
        # False risk-off
        for i in range(5):
            np.random.seed(self.seed + 400 + i)
            paths.append(self.prolonged_false_risk_off(historical_returns))
        
        logger.info(f"Generated {len(paths)} hostile paths")
        
        return paths


if __name__ == "__main__":
    # Quick test
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    
    returns = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)
    regimes = pd.Series(
        np.random.choice(["NEUTRAL", "HOSTILE"], len(dates), p=[0.7, 0.3]),
        index=dates
    )
    
    generator = SyntheticPathGenerator()
    paths = generator.generate_all_hostile_paths(returns, regimes, n_paths_per_type=3)
    
    print(f"\nGenerated {len(paths)} paths:")
    for path in paths[:5]:
        print(f"  {path.name}: {path.description[:50]}...")
