# Kill Switch Decision Tree

This document defines the kill switch hierarchy and responses.

## Overview

Kill switches **OVERRIDE ALL trading logic**. They protect capital in adverse conditions.

```
┌─────────────────────────────────────────────────────────────┐
│                    KILL SWITCH CHECK                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 1. Already Halted?           │
            └──────────────────────────────┘
                    │ YES          │ NO
                    ▼              ▼
            ┌──────────────┐    Continue
            │ STAY HALTED  │
            └──────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 2. Daily Loss > 3%?          │
            └──────────────────────────────┘
                    │ YES          │ NO
                    ▼              ▼
            ┌──────────────┐    Continue
            │ FLATTEN      │
            └──────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 3. Regime Confidence < 0.3?  │
            └──────────────────────────────┘
                    │ YES          │ NO
                    ▼              ▼
            ┌──────────────┐    Continue
            │ FLATTEN      │
            └──────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 4. API Errors ≥ 3?           │
            └──────────────────────────────┘
                    │ YES          │ NO
                    ▼              ▼
            ┌──────────────────────────────┐
            │ FLATTEN + HALT               │
            │ (requires manual reset)      │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 5. Slippage > 15 bps?        │
            │    (3× expected)             │
            └──────────────────────────────┘
                    │ YES          │ NO
                    ▼              ▼
            ┌──────────────────────────────┐
            │ HALT                         │
            │ (requires manual reset)      │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 6. Position Mismatch?        │
            └──────────────────────────────┘
                    │ YES          │ NO
                    ▼              ▼
            ┌──────────────┐    ┌──────────────┐
            │ FLATTEN      │    │ PROCEED      │
            └──────────────┘    │ (Normal)     │
                               └──────────────┘
```

## Actions

| Action | Description |
|--------|-------------|
| **PROCEED** | Continue normal trading |
| **FLATTEN** | Close all positions, cancel orders |
| **HALT** | Flatten + stop all trading until manual reset |

## Thresholds

| Kill Switch | Threshold | Action |
|------------|-----------|--------|
| Daily Loss | > 3% | FLATTEN |
| Confidence | < 0.3 | FLATTEN |
| API Errors | ≥ 3 consecutive | HALT |
| Slippage | > 3× expected | HALT |
| Position Mismatch | > 5% or 1 share | FLATTEN |

## Reset Procedure

To reset a HALT state:

1. Identify root cause
2. Review logs in `logs/trade_audit.csv`
3. Verify broker API is healthy
4. Run: `python -c "from src.monitoring.kill_switches import KillSwitches; ks = KillSwitches(); ks.reset_halt('manual_review')"`
5. Restart trading

## Priority

Kill switches are checked in order. The first trigger takes action.
