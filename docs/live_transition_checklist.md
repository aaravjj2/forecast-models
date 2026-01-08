# Live Transition Checklist

**From Paper to Live Trading**

---

## Pre-Transition Requirements

### 1. Paper Trading Validation
- [ ] **7 consecutive days** of paper trading with no unhandled exceptions
- [ ] **Idempotency verified**: Re-running signals produces zero duplicate orders
- [ ] **Kill-switch tested**: Daily loss and slippage triggers work correctly
- [ ] **Reconciliation**: Broker fills match internal records within tolerance

### 2. Infrastructure Readiness
- [ ] Prometheus metrics exporting correctly
- [ ] Grafana dashboards showing real-time data
- [ ] Alert hooks configured (Slack/Telegram)
- [ ] Audit database persisting all decisions

### 3. Account Setup
- [ ] Live Alpaca account funded
- [ ] API keys rotated (new keys for live)
- [ ] `keys.env` updated with live credentials
- [ ] Emergency contact procedures documented

---

## Transition Steps

### Step 1: Enable Live Mode

```bash
# Set environment variables
export TRADING_MODE=live
export ENABLE_LIVE_TRADING=true
export CONFIRM_LIVE_TRADING=true
```

### Step 2: Start with Minimum Size

Update bot_worker.py or signal sizing to use **50% of calculated position size** for first week.

### Step 3: Monitor First Day

- Watch all orders in real-time
- Compare fills to paper expectations
- Check slippage is within model
- Verify kill-switch would trigger if needed

### Step 4: Gradual Ramp

| Week | Position Size | Monitoring |
|------|--------------|------------|
| 1 | 50% | Continuous |
| 2 | 75% | 3x daily |
| 3-4 | 100% | Daily |
| 5+ | 100% | Weekly review |

---

## Rollback Procedure

If issues detected:

1. **Immediate**: `POST /kill_switch?reason=rollback`
2. **Close positions**: Use Alpaca dashboard
3. **Investigate**: Check logs, reconciliation
4. **Return to paper**: Set `TRADING_MODE=paper`

---

## Post-Transition Monitoring

Daily checklist:
- [ ] Review P&L vs expectation
- [ ] Check slippage metrics
- [ ] Verify no reconciliation discrepancies
- [ ] Confirm kill-switch status is OFF

---

## Emergency Contacts

- Alpaca support: support@alpaca.markets
- Strategy owner: [YOUR EMAIL]
- System admin: [ADMIN EMAIL]
