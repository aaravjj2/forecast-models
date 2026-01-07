# Failure Playbooks

For each failure scenario: Detection → Response → Recovery

---

## 1. Market Regime Change

### Detection
- Regime duration drift > 30% from training
- Correlation with historical patterns < 0.5
- Edge health score < 40 for 2+ weeks

### Response
1. **Do NOT change strategy parameters**
2. Document in stress log
3. Continue paper trading if live stopped
4. Review after 30 days

### Recovery
- If metrics stabilize: Resume normal operation
- If degradation persists: Consider scheduled retrain with new data
- If structural break confirmed: Escalate to retirement review

---

## 2. Model Decay

### Detection
- Entropy collapse (< 0.3)
- Regime hit rate < 50%
- Decay detector alerts

### Response
1. **Do NOT retrain immediately**
2. Verify data pipeline integrity
3. Check for feature calculation errors
4. Compare current feature distributions to training

### Recovery
- If data issue: Fix data, resume
- If distribution shift: Document, monitor 30 days
- If persistent: Schedule model retrain (quarterly cycle)

---

## 3. Broker Failure

### Detection
- 3+ consecutive API errors
- Kill switch triggered
- Order submission failures

### Response
1. System automatically halts (kill switch)
2. Cancel all pending orders manually
3. Verify positions via broker web interface
4. Check Alpaca status page

### Recovery
1. Wait for API stability (minimum 1 hour)
2. Verify account state matches system state
3. Reset kill switch
4. Resume with `--once` flag first

---

## 4. Human Error

### Detection
- Unexpected position in account
- Config mismatch
- Unauthorized parameter change

### Response
1. Emergency flatten if position risk is high
2. Document what happened
3. Rollback config to last known good

### Recovery
1. Review Git history for unauthorized changes
2. Restore from known good commit
3. Re-verify all systems
4. Update access controls if needed

---

## 5. Data Feed Failure

### Detection
- Prices not updating
- Stale timestamps
- Feature calculation errors

### Response
1. Signal generator will return HOLD (fail-safe)
2. Do NOT trade on stale data
3. Check data vendor status

### Recovery
1. Switch to backup data source if available
2. Manual verification of current prices
3. Resume once data is fresh

---

## 6. Slippage Spike

### Detection
- Mean slippage > 15 bps
- Kill switch triggered (slippage breach)
- Fill tracker alerts

### Response
1. Halt trading (automatic)
2. Review market conditions
3. Check order timing

### Recovery
- If temporary (volatility spike): Wait, resume
- If persistent: Consider limit orders
- If structural: Reduce position size

---

## 7. Capital Scaling Issues

### Detection
- Slippage increasing with size
- Partial fills increasing
- Market impact visible

### Response
1. **Do NOT scale further**
2. Reduce back to previous stage
3. Document capacity limit

### Recovery
- Stay at sustainable size
- Update scaling policy with observed limit

---

## General Recovery Checklist

Before resuming after ANY failure:

- [ ] Root cause identified
- [ ] Data pipeline verified
- [ ] Broker API healthy
- [ ] Kill switches reset
- [ ] Config matches last known good
- [ ] Test with `--once` before continuous
- [ ] Document in stress log
