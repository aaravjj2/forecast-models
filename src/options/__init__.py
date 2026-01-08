"""
Options package for options overlay strategies.

Modules:
- data_fetcher: Options chains, IV surface, Greeks
- protective_puts: Tail protection during Risk-Off
- covered_calls: Premium income during Risk-On
- iron_condors: Premium selling in low-vol
- margin_calculator: Margin requirements and limits
- execution_simulator: Realistic execution modeling
"""

from .data_fetcher import OptionsDataFetcher, OptionContract, OptionsChain
from .protective_puts import ProtectivePutStrategy, ProtectivePutPosition
from .covered_calls import CoveredCallStrategy, CoveredCallPosition
from .iron_condors import IronCondorStrategy, IronCondorPosition
from .margin_calculator import MarginCalculator, MarginRequirement, PositionType
from .execution_simulator import OptionsExecutionSimulator, OptionFill, MultiLegFill

__all__ = [
    'OptionsDataFetcher',
    'OptionContract',
    'OptionsChain',
    'ProtectivePutStrategy',
    'ProtectivePutPosition',
    'CoveredCallStrategy',
    'CoveredCallPosition',
    'IronCondorStrategy',
    'IronCondorPosition',
    'MarginCalculator',
    'MarginRequirement',
    'PositionType',
    'OptionsExecutionSimulator',
    'OptionFill',
    'MultiLegFill'
]
