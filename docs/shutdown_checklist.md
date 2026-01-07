# Shutdown Checklist

**When to turn it off.**

Use this checklist to determine if the system should be shut down.

## Immediate Shutdown Triggers

If ANY of these are true, shut down immediately:

- [ ] **Broker API is down** (cannot execute orders)
- [ ] **Account shows unexpected positions** (possible hack/error)
- [ ] **Daily loss exceeds 5%** (beyond kill switch tolerance)
- [ ] **Cannot verify system is running** (silent failure)
- [ ] **Data feed is stale** (prices not updating)

## Considered Shutdown Triggers

Evaluate these - shutdown if multiple are true:

- [ ] **Sharpe < 0 for 60 days** (strategy may be broken)
- [ ] **Slippage consistently > 10 bps** (execution degraded)
- [ ] **2+ P1 incidents in 30 days** (system unreliable)
- [ ] **Regime model accuracy < 55%** (random walk)
- [ ] **Extended market regime shift** (structural break)

## Scheduled Shutdown Windows

The system should be OFF during:

- [ ] System maintenance
- [ ] Broker maintenance windows
- [ ] Major news events (FOMC, earnings if trading single stocks)
- [ ] Extended holidays

## Shutdown Procedure

1. **Flatten positions**
   ```bash
   python -c "
   from src.execution.order_manager import OrderManager
   om = OrderManager()
   om.emergency_flatten('scheduled_shutdown')
   "
   ```

2. **Stop the process**
   ```bash
   # If running continuously
   pkill -f run_paper_trading
   ```

3. **Verify flat**
   ```bash
   python -c "
   from src.execution.alpaca_adapter import AlpacaAdapter
   a = AlpacaAdapter()
   state = a.get_account_state('SPY')
   print(f'Position: {state.position_qty}')
   assert state.position_qty == 0, 'NOT FLAT!'
   print('Verified FLAT')
   "
   ```

4. **Document reason**
   - Add entry to `logs/shutdown_log.txt`
   - Include: timestamp, reason, expected restart

## Restart Checklist

Before restarting after shutdown:

- [ ] Root cause identified (if unplanned)
- [ ] Broker API healthy
- [ ] Data feeds working
- [ ] Kill switches reset if needed
- [ ] Config unchanged from last known good
- [ ] Start with `--once` to verify

## Red Flags to Watch

Signs the strategy may need permanent shutdown:

1. **Consistent underperformance vs buy-and-hold** for 6+ months
2. **Volatility regime no longer predictable** (model degradation)
3. **Market structure change** (new regulations, ETF competition)
4. **Execution impossible** (slippage too high at any size)

If any of these persist, consider:
- Full strategy review
- Model retraining
- Permanent decommission
