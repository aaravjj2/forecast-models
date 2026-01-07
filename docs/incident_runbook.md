# Incident Response Runbook

This document provides step-by-step procedures for handling system incidents.

## Incident Categories

| Category | Severity | Response Time |
|----------|----------|---------------|
| **P0** - System Down | Critical | Immediate |
| **P1** - Trading Halted | High | < 1 hour |
| **P2** - Degraded Performance | Medium | < 4 hours |
| **P3** - Minor Issue | Low | < 24 hours |

---

## P0: System Down (Complete Failure)

### Symptoms
- No logs being generated
- API not responding
- Process crashed

### Response

1. **Verify Status**
   ```bash
   # Check if process is running
   ps aux | grep run_paper_trading
   
   # Check logs
   tail -f logs/paper_trades.csv
   ```

2. **Emergency Flatten (if needed)**
   ```bash
   # Manual position close (if system is down)
   python -c "
   from src.execution.alpaca_adapter import AlpacaAdapter
   a = AlpacaAdapter()
   a.cancel_all_orders()
   a.close_position('SPY')
   print('Positions closed')
   "
   ```

3. **Investigate Root Cause**
   - Check `logs/trade_audit.csv` for last action
   - Check system logs for crashes
   - Verify API credentials are valid

4. **Restart**
   ```bash
   python run_paper_trading.py --once
   ```

---

## P1: Kill Switch Triggered (Trading Halted)

### Symptoms
- Logs show "KILL SWITCH TRIGGERED"
- Trading has stopped
- `is_halted = True`

### Response

1. **Identify Kill Switch Type**
   ```bash
   # Check last audit entries
   tail -20 logs/trade_audit.csv
   ```

2. **For Daily Loss Trigger**
   - Review market conditions
   - Verify loss was real (not data issue)
   - Wait for next trading day (auto-reset)

3. **For API Error Trigger**
   - Check Alpaca API status: https://status.alpaca.markets
   - Test API manually:
     ```bash
     python -c "
     from src.execution.alpaca_adapter import AlpacaAdapter
     a = AlpacaAdapter()
     print(a.is_market_open())
     "
     ```
   - Reset if API is healthy (see P0 step 4)

4. **For Slippage Trigger**
   - Review `reports/slippage_report.json`
   - Check if market was volatile
   - Adjust thresholds if needed

5. **Manual Reset**
   ```bash
   python -c "
   from src.monitoring.kill_switches import KillSwitches
   ks = KillSwitches()
   ks.reset_halt('incident_response')
   print('Halt reset')
   "
   ```

---

## P2: Degraded Performance

### Symptoms
- Higher than expected slippage
- Orders taking longer to fill
- Partial fills

### Response

1. **Check Slippage Stats**
   ```bash
   python -c "
   from src.execution.fill_tracker import FillTracker
   ft = FillTracker()
   print(ft.get_slippage_stats())
   "
   ```

2. **Compare to Baseline**
   - Expected: 5 bps mean
   - If > 10 bps: Investigate

3. **Possible Causes**
   - High market volatility
   - Low liquidity
   - Timing of order submission

4. **Actions**
   - Consider switching to limit orders
   - Adjust position sizing
   - Document in weekly reconciliation

---

## P3: Minor Issues

### Examples
- Log file corrupted
- Missed single execution
- Config drift

### Response

1. **Document Issue**
2. **Check logs**
3. **Fix in next scheduled maintenance**

---

## Communication Template

For any P0 or P1 incident, use this template:

```
INCIDENT REPORT
===============
Date/Time: [UTC]
Severity: [P0/P1/P2/P3]
Status: [Investigating/Mitigated/Resolved]

Summary:
[Brief description]

Impact:
[What was affected]

Root Cause:
[If known]

Actions Taken:
1. [Action 1]
2. [Action 2]

Next Steps:
- [Step 1]
- [Step 2]
```

---

## Contacts

| Role | Contact |
|------|---------|
| System Owner | (define) |
| Broker Support | Alpaca support |

---

## Post-Incident Review

After any P0 or P1:

1. Create incident report
2. Identify root cause
3. Document lessons learned
4. Update runbook if needed
5. Implement preventive measures
