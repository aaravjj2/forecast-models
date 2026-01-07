# Maintenance Rules

Long-term maintenance schedule and conditions.

---

## Review Cadence

| Review | Frequency | Duration | Owner |
|--------|-----------|----------|-------|
| **Daily Check** | Every trading day | 2 min | Automated |
| **Weekly Reconciliation** | Friday | 10 min | Automated + glance |
| **Monthly Report** | 1st of month | 30 min | Manual review |
| **Quarterly Strategy Review** | Every 3 months | 2 hours | Full analysis |
| **Annual Audit** | Yearly | 1 day | Complete review |

---

## Daily Check (Automated)

- [ ] Kill switches status: healthy
- [ ] No unclosed orders
- [ ] Equity within expected range
- [ ] Log files updating

**If any fail**: Check kill switch status, review logs.

---

## Weekly Reconciliation

- [ ] Compare paper PnL to backtest expectations
- [ ] Review slippage distribution
- [ ] Check edge health metrics
- [ ] Update stress log if events occurred

Generate report: `python -m src.execution.reconciliation`

---

## Monthly Report

Review:
1. Edge attribution breakdown
2. Regime hit rate trend
3. Stability scorecard
4. Decay detection alerts

**Decision**: Is the system behaving as expected?
- If YES: Continue
- If NO: Document deviation, monitor

---

## Quarterly Strategy Review

Full analysis:
1. Rolling Sharpe vs backtest
2. Drawdown comparison
3. Regime persistence analysis
4. Model confidence distribution

**Decisions**:
- Capital scaling: Advance or rollback?
- Model: Retrain needed?
- Parameters: Any changes required?

> **Default answer**: No changes needed.

---

## Conditions for Retraining

Retrain the model ONLY if:

1. **Scheduled cycle** (every 6 months) AND metrics support it
2. **Decay detector alerts persist** for 60+ days
3. **Regime hit rate** drops below 45%

> [!CAUTION]
> Retraining is rarely needed and often harmful.
> Most "decay" is noise. Wait and observe.

### Retrain Procedure

1. Freeze current model
2. Collect last 2 years of data
3. Train new model with same hyperparameters
4. Paper trade new model for 30 days
5. Compare to current model
6. Only deploy if improvement is significant (Sharpe +0.2)

---

## Conditions for Retirement

Consider permanent shutdown if:

- [ ] Sharpe < 0 for 6+ months
- [ ] Regime prediction accuracy < random (50%)
- [ ] Market structure fundamentally changed
- [ ] Better capital deployment opportunity exists

### Retirement Procedure

1. Document decision and rationale
2. Flatten all positions
3. Archive codebase
4. Final report generation
5. No regrets (this is correct behavior)

---

## Parameter Changes

Any parameter change requires:

1. Written justification
2. Backtest with new parameter
3. Git PR
4. Review approval
5. Gradual rollout

> **Default**: Parameters should NOT change.

---

## Emergency Contacts

| Role | Responsibility |
|------|----------------|
| System Owner | Final decisions |
| Broker Support | Alpaca issues |

---

## Final Note

> The best maintenance is no maintenance.
> 
> If the system needs constant attention, something is wrong with the design, not the execution.
