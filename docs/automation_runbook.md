# Automation Runbook

**Volatility-Gated Strategy Execution System**

---

## Quick Start

```bash
# Paper mode (default)
cd /home/aarav/Forecast\ models/ml_research_pipeline
python -m src.automation.bot_worker --symbol SPY --once

# Start executor service
python -m src.automation.executor_service --port 8000
```

---

## Startup Checklist

- [ ] Verify `keys.env` contains valid Alpaca credentials
- [ ] Check Redis is running (if using queue): `redis-cli ping`
- [ ] Verify market is open: US market hours 9:30 AM - 4:00 PM ET
- [ ] Check kill switch is OFF: `GET /health` → `kill_switch_active: false`

---

## Shutdown Procedure

1. **Stop accepting new signals**: `POST /kill_switch?reason=manual_shutdown`
2. **Wait for pending orders**: Check `GET /health` → `orders_placed == orders_filled`
3. **Stop executor**: `Ctrl+C` or `kill -TERM <pid>`
4. **Stop bot worker**: `Ctrl+C` on scheduler

---

## Manual Overrides

### Force Kill Switch
```bash
curl -X POST "http://localhost:8000/kill_switch?reason=emergency"
```

### Reset Kill Switch
Set environment variable first:
```bash
export ALLOW_KILL_SWITCH_RESET=true
curl -X POST "http://localhost:8000/reset_kill_switch"
```

### Force Position Close
Not implemented in automation - use Alpaca dashboard directly.

---

## Log Interpretation

| Log Level | Meaning |
|-----------|---------|
| `INFO: Order X filled` | Normal fill |
| `WARNING: Rate limit exceeded` | Slow down API calls |
| `WARNING: Circuit breaker OPEN` | 3+ consecutive failures |
| `CRITICAL: KILL SWITCH TRIGGERED` | Trading halted - investigate immediately |

---

## Metrics (Prometheus)

Metrics available at `http://localhost:9090/metrics`:

- `volatility_gated_orders_total{symbol, side, status}` - Order counts
- `volatility_gated_slippage_bps{symbol, side}` - Slippage distribution
- `volatility_gated_daily_pnl_usd` - Daily P&L
- `volatility_gated_kill_switch_active` - Kill switch status (0/1)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Orders not filling | Circuit breaker open | Check `GET /health`, wait 5 min for reset |
| Kill switch triggered | Daily loss or slippage breach | Investigate cause, reset manually |
| No signals received | Bot worker not running | Restart with `--once` flag |
| API errors | Rate limit | Reduce `rate_limit_per_minute` |
