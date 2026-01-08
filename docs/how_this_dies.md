# How This Dies

**One-Page Failure Mode Documentation**

> [!CAUTION]
> This document exists to make failure explicit. If you find yourself in one of these scenarios, the system has failed and you must take action.

---

## 1. Maximum Drawdown Scenarios

### Flash Crash (Intraday)
- **Trigger**: Market drops >10% in minutes
- **System Response**: Kill-switch may not trigger fast enough
- **Max Expected Loss**: 15-20% before safety triggers
- **Recovery**: Automatic halt, manual position review

### Overnight Gap
- **Trigger**: Major event after hours (earnings, geopolitics)
- **System Response**: Cannot exit overnight positions
- **Max Expected Loss**: Unlimited for long positions
- **Recovery**: Accept loss, review position sizing

### Regime Model Failure
- **Trigger**: Volatility spikes without historical precedent
- **System Response**: Model outputs ~50% confidence (noise)
- **Max Expected Loss**: Hold through crash (buy-and-hold equivalent)
- **Recovery**: Discretionary override to cash

---

## 2. Psychological Stress Scenarios

### Extended Drawdown
- **Trigger**: 3+ months of underperformance
- **Symptoms**: Urge to "tweak" parameters, add models
- **Response**: DO NOTHING. Follow the process.
- **Escalation**: If DD > 20%, halt and review

### Missing a Rally
- **Trigger**: System in flat while market rallies 20%+
- **Symptoms**: FOMO, desire to override signals
- **Response**: The edge is crash avoidance, not rally capture
- **Escalation**: Review expectancy, not returns

### Slippage Pain
- **Trigger**: Fills consistently worse than expected
- **Symptoms**: Distrust of execution layer
- **Response**: Check reconciliation reports, verify data
- **Escalation**: If slippage > 3x expected, halt

---

## 3. System Death Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Cumulative Drawdown | > 25% | Permanent halt, post-mortem |
| Sharpe Ratio (6mo) | < 0 | Reduce capital 50% |
| Regime Accuracy | < 55% for 3 months | Model retraining required |
| Reconciliation Failures | > 10% divergence | System bug, halt |
| Kill-Switch Triggers | 3+ in 30 days | Review edge validity |

---

## 4. What NOT To Do

> [!WARNING]
> These actions will destroy the edge faster than any market event.

1. **Parameter Optimization**: Curve-fitting to recent data
2. **Adding Models**: Complexity = fragility
3. **Overriding Signals**: One exception becomes the rule
4. **Chasing Returns**: The edge is defensive, not offensive
5. **Ignoring Slippage**: 10 bps is the cost floor, not ceiling

---

## 5. Recovery Procedure

If system enters failure state:

1. **HALT**: Stop all trading immediately
2. **RECONCILE**: Run full reconciliation report
3. **DIAGNOSE**: Use `docs/diagnostic_checklist.md`
4. **DOCUMENT**: Log exactly what happened
5. **WAIT**: Minimum 7 days before any changes
6. **REVIEW**: Independent review of any proposed fixes

---

## 6. Final Truth

The system is designed to **avoid catastrophic loss**, not to generate maximum returns.

If it dies, it should die slowly (drawdown over months) rather than suddenly (blown account).

If you find yourself checking P&L hourly, you have already lost.

---

**Last Updated**: 2026-01-07
**Owner**: [STRATEGY OWNER]
