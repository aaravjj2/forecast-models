# "What Broke?" Diagnostic Checklist

When performance degrades, use this checklist to identify the cause.

## Step 1: Confirm the Problem

- [ ] Is the degradation real or noise?
  - Check: Is drawdown > 2 standard deviations from backtest?
  - Check: Has degradation persisted > 5 trading days?
- [ ] Is this a market-wide event or strategy-specific?
  - Compare: Strategy vs Buy & Hold during same period

If degradation is within normal variance, **stop here**. Do not investigate further.

---

## Step 2: Attribution Analysis

Run: `python -m src.analytics.edge_attribution`

### Questions to Answer

| Question | Where to Look |
|----------|---------------|
| Did we lose money on exposed days? | `market_beta` component |
| Did we fail to avoid crashes? | `vol_avoidance` component |
| Did entry/exit timing decay? | `timing_contribution` component |
| Are costs eating returns? | `cost_drag` vs historical |
| Is slippage worse than expected? | `slippage_impact` vs baseline |

### Primary Edge Check

- [ ] Is `vol_avoidance` still positive?
  - If NO → Regime model may be broken
  - If YES → Edge intact, look elsewhere

---

## Step 3: Edge Health Check

Run: `python -m src.analytics.edge_health`

### Critical KPIs

| KPI | Action Threshold |
|-----|------------------|
| Regime Hit Rate | < 50% → INVESTIGATE |
| Crash Avoidance Delta | < 0% → INVESTIGATE |
| Health Score | < 40 → ALERT |

---

## Step 4: Decay Detection

Run: `python -m src.analytics.decay_detector`

### Non-Performance Signals

| Signal | What It Means |
|--------|---------------|
| Entropy Collapse | Model over-confident → possible overfit |
| Probability Clustering | Predictions stuck at extremes |
| Duration Drift | Regime persistence changed |
| Correlation Drift | Market structure may have changed |

---

## Step 5: Root Cause Categories

### A. Data Problem
- [ ] Is price data stale or incorrect?
- [ ] Has data vendor changed format?
- [ ] Are features calculating correctly?

### B. Model Decay
- [ ] Has volatility regime accuracy dropped?
- [ ] Is the model predicting outside training distribution?
- [ ] Has feature importance shifted dramatically?

### C. Market Structure Change
- [ ] New regulatory environment?
- [ ] VIX behavior fundamentally different?
- [ ] Correlation regimes shifted?

### D. Execution Problem
- [ ] Slippage > 2x expected?
- [ ] Partial fills increasing?
- [ ] Broker latency increased?

---

## Step 6: Decision Matrix

| Root Cause | Action |
|------------|--------|
| **Data Problem** | Fix data, do NOT touch model |
| **Model Decay** | Document, monitor, consider scheduled retrain |
| **Market Change** | ACCEPT. This is risk. Do not chase. |
| **Execution Problem** | Fix execution, do NOT touch model |

---

## Step 7: Documentation

After investigation, document:

1. **Date**: When degradation was noticed
2. **Duration**: How long it persisted
3. **Root Cause**: From categories above
4. **Action Taken**: What (if anything) was done
5. **Outcome**: Did action resolve issue?

Add to `docs/stress_log.md`.

---

## Red Flags (Immediate Attention)

> [!CAUTION]
> If ANY of these are true, escalate immediately:
> - Daily loss > 5%
> - Regime hit rate < 40%
> - 3+ consecutive failed predictions
> - Kill switch triggered

---

## Remember

> **This is observation, not optimization.**
> 
> Finding the cause does NOT mean "fixing" it.
> 
> Most degradation is noise. Act only on structural breaks.
