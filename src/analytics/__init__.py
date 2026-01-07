"""
Analytics package for edge attribution, health monitoring, and decay detection.

Modules:
- edge_attribution: PnL decomposition into components
- edge_health: True KPIs (regime hit rate, crash avoidance, etc.)
- decay_detector: Early-warning signals for edge degradation
- regime_persistence: Duration and transition analysis
- stability_metrics: Rolling performance stability
"""

from .edge_attribution import EdgeAttribution, DailyAttribution, AttributionSummary
from .edge_health import EdgeHealth, EdgeHealthSnapshot
from .decay_detector import DecayDetector, DecayStatus, DecayAlert, AlertSeverity
from .regime_persistence import RegimePersistenceAnalyzer, RegimePersistence, PersistenceReport
from .stability_metrics import StabilityMetrics, StabilitySnapshot, StabilityReport

__all__ = [
    'EdgeAttribution',
    'DailyAttribution',
    'AttributionSummary',
    'EdgeHealth',
    'EdgeHealthSnapshot',
    'DecayDetector',
    'DecayStatus',
    'DecayAlert',
    'AlertSeverity',
    'RegimePersistenceAnalyzer',
    'RegimePersistence',
    'PersistenceReport',
    'StabilityMetrics',
    'StabilitySnapshot',
    'StabilityReport'
]
