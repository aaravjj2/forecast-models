# Robustness Testing Guide

## Overview

This comprehensive testing suite validates your ML pipeline to ensure it has **real edge**, not just overfitting or data leakage.

## PHASE 1: Sanity & Robustness Checks (DO THIS FIRST)

### Test 1: Confidence Threshold Sweep
**Purpose:** Kill the illusion - lower confidence threshold to see real performance

**What it does:**
- Tests thresholds: 0.7 → 0.6 → 0.55 → 0.5 → 0.45
- Records: trades count, accuracy, Sharpe, max drawdown

**What you want to see:**
- ✅ Accuracy falls **slowly** (not collapses)
- ✅ Trade count rises
- ✅ PnL stabilizes

**Red flags:**
- ❌ Accuracy collapses immediately → overfitting
- ❌ Sharpe drops dramatically → edge was fake

**Run:**
```bash
python robustness_tests.py --phase 1
```

### Test 2: Shuffle Test (Anti-Overfitting)
**Purpose:** Critical test most people skip - destroys signal to check for leakage

**What it does:**
- Randomly shuffles labels
- Re-runs entire pipeline
- Evaluates on real labels

**Expected:**
- ✅ Accuracy ≈ 50% (random)
- ✅ No profitable trades
- ✅ Sharpe ≈ 0

**Red flags:**
- ❌ Still high accuracy → **data leakage exists!**
- ❌ Still profitable → model is cheating

**Run:** Included in Phase 1

### Test 3: Time-Shift Test
**Purpose:** Check if model is cheating via timing

**What it does:**
- Shifts features forward/backward by 1-2 days
- Re-runs backtest

**Expected:**
- ✅ Performance degrades with shifts
- ✅ Baseline (no shift) performs best

**Red flags:**
- ❌ Performance doesn't degrade → model using future information

**Run:** Included in Phase 1

## PHASE 2: Expand Universe (Reality Check)

### Test 4: Multi-Stock Test
**Purpose:** Does the pattern hold across different stocks?

**What it does:**
- Runs pipeline on: AAPL, MSFT, NVDA, AMZN, META, TSLA
- Compares behavior (not just accuracy)

**Key metrics:**
- Similar abstention/trade ratios
- Similar confidence distributions
- Consistent performance patterns

**Red flags:**
- ❌ Works only on AAPL
- ❌ Collapses on others
- ❌ Wildly different behavior

**Green flags:**
- ✅ Similar patterns across stocks
- ✅ Consistent abstention logic
- ✅ Reasonable performance everywhere

**Run:**
```bash
python robustness_tests.py --phase 2
```

### Test 5: Cross-Stock Generalization
**Purpose:** Hard but important - train on some, test on unseen

**What it does:**
- Trains on: AAPL, MSFT, NVDA, AMZN
- Tests on: META, TSLA (unseen)

**This is brutal but real.** Most finance models fail here.

**Expected:**
- Performance degrades (but not collapses)
- Still some signal remains

**Red flags:**
- ❌ Complete collapse → no generalization
- ❌ Negative performance → overfitting

**Run:** Included in Phase 2

## PHASE 3: Stress Ensemble Logic

### Test 6: Disable Components
**Purpose:** Identify which component actually adds signal

**What it does:**
- Runs with: no sentiment, no rules, no LGBM, no XGB, XGB only, simple average

**Goal:**
- Identify which component helps
- If removing a model **improves** results → it was hurting you

**Run:**
```bash
python robustness_tests.py --phase 3
```

### Test 7: Meta-Model Explainability
**Purpose:** Understand which model is trusted when

**Action needed:** Log meta-ensemble weights per trade

**Questions to answer:**
- Which model was trusted?
- In which regime?
- Patterns like: "Sentiment dominates during earnings weeks"

**Status:** Framework ready, needs logging implementation

## PHASE 4: Reality Alignment

### Test 8: Minimum Trade Frequency
**Purpose:** Force model to take responsibility

**What it does:**
- Introduces minimum trade frequency constraint
- Forces model to trade even when uncertain

**Expected:**
- Accuracy will drop (that's good - realistic)
- More trades
- More realistic performance

**Run:**
```bash
python test_minimum_trade_frequency.py
```

### Test 9: Transaction Costs
**Purpose:** Add real-world costs

**What it does:**
- Adds 5-10 bps per trade
- Small slippage noise

**Expected:**
- PnL should decrease but not collapse
- Sharpe should remain positive

**Red flags:**
- ❌ PnL collapses → edge was fake
- ❌ Negative after costs → not tradeable

**Run:** Included in Phase 4

## PHASE 5: Decision Framework

After all tests, answer honestly:

### Does accuracy stay >52-55% with:
- ✅ 100+ trades
- ✅ Across multiple stocks
- ✅ With transaction costs?

**If YES → you have something real** 🎉
**If NO → still valuable research, but no edge yet** 📊

## Running All Tests

```bash
# Run everything
python robustness_tests.py --all

# Run specific phase
python robustness_tests.py --phase 1  # Sanity checks
python robustness_tests.py --phase 2  # Multi-stock
python robustness_tests.py --phase 3  # Ensemble stress
python robustness_tests.py --phase 4  # Reality alignment
```

## Results Location

All results saved to: `robustness_test_results/`

- `*_confidence_threshold_test.csv` - Threshold sweep results
- `*_shuffle_test.json` - Shuffle test results
- `*_time_shift_test.csv` - Time-shift results
- `multi_stock_test.csv` - Multi-stock comparison
- `cross_stock_generalization.json` - Cross-stock results
- `*_disable_components_test.csv` - Component ablation
- `*_transaction_costs_test.csv` - Cost impact

## Interpreting Results

### Good Signs ✅
- Accuracy degrades slowly with lower thresholds
- Shuffle test shows ~50% accuracy
- Performance degrades with time shifts
- Similar patterns across stocks
- Positive returns after transaction costs

### Bad Signs ❌
- Accuracy collapses immediately
- Shuffle test still shows high accuracy
- Time shifts don't affect performance
- Works only on one stock
- Negative returns after costs

## Next Steps

1. **Run Phase 1 first** - This will tell you if you have fundamental issues
2. **If Phase 1 passes** - Move to Phase 2 (multi-stock)
3. **If Phase 2 passes** - You might have real edge!
4. **Run Phase 3 & 4** - Stress test and add reality
5. **Make decision** - Based on Phase 5 criteria

Good luck! 🚀

