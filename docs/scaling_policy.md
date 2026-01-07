# Capital Scaling Policy

This document defines how capital increases safely without destroying the edge.

## Core Principles

> **This is RISK alpha, not RETURN alpha.**

1. Capacity is limited by liquidity, overnight gaps, and execution friction
2. Scaling too fast kills Sharpe first, then capital
3. Conservative is correct

## Scaling Schedule

**Step-function scaling (NOT continuous)**

| Stage | Capital | Requirements | Rollback Trigger |
|-------|---------|--------------|------------------|
| **0: Paper** | $0 (simulated) | Model trained | N/A |
| **1: Seed** | $10,000 | Paper validation complete (5+ days) | Any execution failure |
| **2: Small** | $25,000 | 3 months live, DD < 10%, no halts | DD > 15% or halt |
| **3: Medium** | $50,000 | 6 months live, Sharpe > 0.8 stable | Sharpe < 0.5 for 30 days |
| **4: Target** | $100,000 | 12 months, all metrics stable | Any governance violation |

## Stage Advancement Criteria

### Paper → Stage 1 (Seed)
- [ ] Paper trading for minimum 5 trading days
- [ ] No execution failures
- [ ] Slippage within bounds (±1 std)
- [ ] Strategy "boring" (no excitement)

### Stage 1 → Stage 2 (Small)
- [ ] 3 months of live trading
- [ ] Maximum drawdown < 10%
- [ ] No kill switch activations (except HOLD)
- [ ] Slippage stable within model

### Stage 2 → Stage 3 (Medium)
- [ ] 6 months cumulative live trading
- [ ] Sharpe ratio > 0.8 (rolling 6 month)
- [ ] Weekly reconciliation clean
- [ ] No discretionary overrides

### Stage 3 → Stage 4 (Target)
- [ ] 12 months cumulative live trading
- [ ] All metrics stable
- [ ] Quarterly review passed
- [ ] No governance violations

## Rollback Rules

If any of these occur, immediately reduce to previous stage:

| Condition | Action |
|-----------|--------|
| Sharpe < 0.5 for 30 consecutive days | Reduce 1 stage |
| Drawdown > 15% | Reduce 1 stage |
| Kill switch HALT triggered | Reduce 1 stage |
| 2+ P1 incidents in 30 days | Reduce 1 stage |
| Governance violation | Reduce to Paper |

## Recovery After Rollback

To advance again after rollback:

1. Identify and fix root cause
2. Wait 30 days at lower stage
3. Pass all stage criteria again

## Capacity Estimation

### Position Size Limits

| Capital | Max Position (% of ADV) | Estimated Slippage |
|---------|------------------------|-------------------|
| $10K | 0.01% | ~3 bps |
| $25K | 0.025% | ~4 bps |
| $50K | 0.05% | ~5 bps |
| $100K | 0.1% | ~7 bps |
| $500K | 0.5% | ~15 bps |
| $1M+ | >1% | **DEGRADED** |

### Hard Limits

- **Soft Cap**: Slippage > 5 bps average
- **Hard Cap**: Slippage > 15 bps (strategy breaks)
- **Absolute Max**: $500K for SPY (0.5% ADV)

## Review Schedule

| Review Type | Frequency | Owner |
|-------------|-----------|-------|
| Weekly Reconciliation | Weekly | Automated |
| Monthly Metrics Review | Monthly | Manual |
| Quarterly Strategy Review | Quarterly | Manual |
| Stage Advancement | As needed | Manual |
