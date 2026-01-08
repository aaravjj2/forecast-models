"""
Validation package for adversarial testing.

Modules:
- null_strategies: Baseline comparisons (Random, Always-Long, etc.)
- permutation_tests: Block bootstrap permutation testing
- synthetic_paths: Hostile market path generation
- portfolio_stress: Multi-asset stress testing
"""

from .null_strategies import NullStrategies, NullResult
from .permutation_tests import PermutationTester, PermutationResult
from .synthetic_paths import SyntheticPathGenerator, SyntheticPath
from .portfolio_stress import PortfolioStressTester, PortfolioStressSummary

__all__ = [
    'NullStrategies',
    'NullResult',
    'PermutationTester',
    'PermutationResult',
    'SyntheticPathGenerator',
    'SyntheticPath',
    'PortfolioStressTester',
    'PortfolioStressSummary'
]
