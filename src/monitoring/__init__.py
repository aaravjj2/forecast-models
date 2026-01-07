"""
Monitoring package for operational safety.

Modules:
- kill_switches: Mandatory safety shutdown conditions
- audit_logger: Complete decision trail logging
"""

from .kill_switches import KillSwitches, KillSwitchConfig, KillSwitchState, KillSwitchResult, KillSwitchType
from .audit_logger import AuditLogger, AuditRecord

__all__ = [
    'KillSwitches',
    'KillSwitchConfig',
    'KillSwitchState',
    'KillSwitchResult',
    'KillSwitchType',
    'AuditLogger',
    'AuditRecord'
]
