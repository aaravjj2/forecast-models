# Governance Document

This document defines who can change what, and how.

## Core Principle

> **No discretionary overrides allowed.**
> 
> The system operates by rules, not intuition.

## Change Matrix

| Action | Approval Required | Process |
|--------|-------------------|---------|
| **Emergency Shutdown** | None (single operator) | Execute immediately |
| **Config Parameter Change** | Git PR + 1 review | PR → Review → Merge → Deploy |
| **Threshold Adjustment** | Written justification + 1 review | Document → PR → Review |
| **Execution Logic Change** | Full backtest re-run + 2 reviews | Backtest → Document → PR → 2 Reviews |
| **Model Retrain** | Scheduled or after failure | Trigger → Train → Validate → Deploy |
| **Stage Advancement** | All criteria met + manual approval | Checklist → Review → Advance |

## Roles

| Role | Permissions |
|------|-------------|
| **Operator** | Execute shutdown, view logs, generate reports |
| **Developer** | Create PRs, modify code, run backtests |
| **Reviewer** | Approve PRs, authorize changes |

## Mandatory Review Intervals

| Review | Frequency | Owner |
|--------|-----------|-------|
| **Weekly Reconciliation** | Every Friday | Automated |
| **Monthly Performance** | 1st of month | Operator |
| **Quarterly Strategy** | Every 3 months | Developer + Reviewer |
| **Annual Full Audit** | Yearly | All |

## Prohibited Actions

The following are NEVER allowed:

1. ❌ Hardcoding API credentials
2. ❌ Modifying production config directly (bypass Git)
3. ❌ Overriding kill switches
4. ❌ Deploying untested code
5. ❌ Scaling without meeting criteria

## Emergency Procedures

### System Shutdown

Any single operator can shut down the system at any time:

```bash
# Emergency flatten and halt
python -c "
from src.execution.order_manager import OrderManager
om = OrderManager()
om.emergency_flatten('manual_shutdown')
"
```

No approval required. Document reason after.

### Kill Switch Override

**KILL SWITCHES CANNOT BE OVERRIDDEN IN PRODUCTION.**

To temporarily disable for debugging (PAPER ONLY):

```bash
# This is logged
python -c "
from src.monitoring.kill_switches import KillSwitches
ks = KillSwitches()
ks.state.is_halted = False  # Paper only
print('WARNING: Kill switch temporarily disabled')
"
```

## Audit Trail

All changes are tracked:

1. **Code**: Git history
2. **Config**: Git history
3. **Decisions**: `logs/trade_audit.csv`
4. **Incidents**: Incident reports

## Violation Response

| Violation | Response |
|-----------|----------|
| 1st minor | Warning + documentation |
| 2nd minor | Reduced permissions |
| Major | Immediate halt + review |
| Critical | Full shutdown + investigation |
