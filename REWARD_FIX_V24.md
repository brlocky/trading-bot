# Reward Function Fix - v24

## Problem Diagnosis

### Symptoms
- Model trained successfully (65k steps, good metrics)
- But: 96% HOLD actions, only 4% trading
- When trading, slight preference for SHORT (8.6% vs 6.5% prob)

### Root Cause Analysis
1. **Risk-Reward Imbalance**:
   - Opening position: +2.0 reward
   - Hitting SL: -15 to -30 penalty
   - Expected value: NEGATIVE → Model learned not to trade

2. **No Cost for Inaction**:
   - Staying HOLD has zero penalty
   - Trading has risk → Optimal strategy = HOLD forever

3. **Training Data Was Balanced**:
   - 49.4% up bars, 49.9% down bars
   - +6.05% total return over 32k samples
   - Test data: +1.19% return, 50.3% up bars
   - **Not a data bias issue**

4. **Reward Function Was Symmetric**:
   - LONG and SHORT treated identically
   - Broker's unrealized PnL calculation correct
   - **Not a code bias issue**

### Key Finding
**Model optimized correctly** - it found that HOLD is safer than trading given the reward structure!

## Solution: Reward Function v24

### Changes Made

| Component | v23 (Old) | v24 (New) | Reason |
|-----------|-----------|-----------|---------|
| **Exploration Bonus** | +2.0 | +10.0 | 5x increase to encourage action |
| **SL Penalty** | -15 to -30 | -10 to -20 | 33% reduction to reduce fear |
| **Idle Penalty** | 0.0 | -0.5/step | Discourage excessive HOLDing |
| **TP Rewards** | Unchanged | Unchanged | Already good |
| **Balance Rewards** | Unchanged | Unchanged | Primary goal intact |
| **Redundant Action** | -50.0 | -50.0 | Keep strong penalty |

### Expected Outcomes

1. **Increased Trading Activity**:
   - HOLD should drop from 96% → 40-60%
   - LONG/SHORT combined should be 30-50%

2. **Balanced Directionality**:
   - LONG and SHORT probabilities should equalize
   - Model will trade in both directions based on market

3. **Better Risk-Taking**:
   - Exploration bonus (+10) now covers ~50% of average SL (-20)
   - Net risk-reward more attractive

### Risk-Reward Math

**Before (v23):**
```
Open position: +2.0
Avg SL hit:    -22.5 (midpoint of -15 to -30)
Net expected:  -20.5 → DON'T TRADE!
```

**After (v24):**
```
Open position: +10.0
Avg SL hit:    -15.0 (midpoint of -10 to -20)
Idle penalty:  -0.5/step (if don't trade)
Net expected:  Better to try trading
```

## Testing Plan

1. **Retrain with v24**:
   - Use same Preset 1 (65k steps)
   - Monitor action distribution during training
   - Check if HOLD% decreases over time

2. **Validate Action Distribution**:
   ```bash
   python test_action_distribution.py
   ```
   - Target: HOLD ~50%, LONG ~20%, SHORT ~20%, CLOSE ~10%

3. **Check Performance**:
   - Run full backtest on unseen data
   - Ensure trades are profitable
   - Verify LONG/SHORT balance

## Files Modified

- `src/environments/simple_trading_env.py`: Updated `calculate_reward()` method (v23 → v24)
- `RL_BACK.ipynb`: Added diagnostic markdown cell
- `test_action_distribution.py`: Created for action probability analysis

## Next Steps

1. ✅ Reward function updated
2. ⏳ Retrain model with v24
3. ⏳ Test action distribution
4. ⏳ Validate on unseen data
5. ⏳ Compare performance vs v23
