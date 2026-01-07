# Strategy Doctrine

**This is the single source of truth.**

---

## What This System Does

The **Volatility-Gated Long Exposure** strategy:

1. **Goes LONG** when volatility regime is Risk-On (low volatility, stable markets)
2. **Goes FLAT** when volatility regime is Risk-Off (high volatility, unstable markets)
3. **Executes at market open** based on previous day's signal (T+1 execution)

**That's it.** There is no other logic.

---

## What This System Does NOT Do

- ❌ Does NOT predict market direction
- ❌ Does NOT pick stocks
- ❌ Does NOT use leverage beyond position sizing
- ❌ Does NOT trade intraday
- ❌ Does NOT adapt to news in real-time
- ❌ Does NOT optimize parameters based on recent performance
- ❌ Does NOT override regime signals with discretion

---

## Why It Works (The Edge)

**Volatility Avoidance**, not Return Maximization.

The edge exists because:
1. Volatility clusters and persists (well-documented phenomenon)
2. Most damage happens during volatility spikes
3. Being flat during crashes saves more than the upside you miss

**This is defense, not offense.**

---

## When To Shut It Down

### Temporary Shutdown
- Daily loss > 3% → Automatic (kill switch)
- Broker API failure → Automatic (kill switch)
- Slippage > 3× expected → Automatic (kill switch)

### Permanent Shutdown Consideration
- Sharpe < 0 for 60+ consecutive days
- Regime hit rate < 45% for 3 months
- Structural market change makes volatility unpredictable
- You feel the urge to "fix" or "improve" it

---

## The Philosophy

> "This system is designed to be boring."
> 
> If it's exciting, something is wrong.

The goal is **survival**, not maximization. A strategy that survives longer compounds more than one that optimizes for short-term returns.

---

## Who Can Change What

| Action | Approval |
|--------|----------|
| Emergency shutdown | Anyone (no approval) |
| Config parameter | Git PR + 1 review |
| Execution logic | Full backtest + 2 reviews |
| Model retrain | Scheduled or explicit decay trigger |

---

## How To Explain It in 5 Minutes

"We go long when markets are calm and step aside when they're volatile.
We don't predict direction. We avoid damage.
The edge is the crashes we miss, not the rallies we catch.
It's boring by design."

---

## Final Reminder

> **No heroics required.**
> 
> If you can pause this system for 6 months and return to find nothing needed attention, it's working correctly.
