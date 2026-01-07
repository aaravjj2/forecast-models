# Live Stress Log

**Purpose**: Document strategy behavior during live market stress events.

---

## Template for Each Event

Copy this template for each stress event:

```markdown
### Event: [Name]
**Date**: YYYY-MM-DD
**Type**: [Macro Shock | Volatility Spike | News Gap | Sideways Chop]

#### Market Conditions
- VIX: [level]
- Market Return: [%]
- Duration: [days]

#### Strategy Behavior
- Regime State: [HOSTILE | NEUTRAL | FAVORABLE]
- Position: [LONG | FLAT]
- Signal Generated: [ENTER | EXIT | HOLD]

#### Outcome
- Strategy Return: [%]
- Buy & Hold Return: [%]
- Crash Avoidance: [Yes/No]
- Kill Switch Triggered: [Yes/No]

#### Notes
[Free-form observations]
```

---

## Logged Events

### Event: [Template - Delete this]
**Date**: YYYY-MM-DD
**Type**: Example

#### Market Conditions
- VIX: 20
- Market Return: -2%
- Duration: 1 day

#### Strategy Behavior
- Regime State: HOSTILE
- Position: FLAT
- Signal Generated: EXIT

#### Outcome
- Strategy Return: 0%
- Buy & Hold Return: -2%
- Crash Avoidance: Yes
- Kill Switch Triggered: No

#### Notes
System performed as expected.

---

## Summary Statistics

| Event Type | Count | Win Rate | Avg Protection |
|------------|-------|----------|----------------|
| Macro Shock | 0 | - | - |
| Volatility Spike | 0 | - | - |
| News Gap | 0 | - | - |
| Sideways Chop | 0 | - | - |

---

*Last Updated: [Date]*
