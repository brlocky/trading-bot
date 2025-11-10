# Feature Extraction Rebalancing - v2

## Problem Analysis

### Feature Activation Imbalance
From testing the trained model, we discovered severe imbalance in feature usage:

| Encoder | Activation | Relative Strength | Issue |
|---------|-----------|-------------------|--------|
| Session_Encoder | 0.3990 | **12.3x** | Too dominant |
| Account_Encoder | 0.1568 | 4.8x | OK |
| Position_Encoder | 0.1397 | 4.3x | OK |
| Trend_Encoder | 0.1242 | 3.8x | OK |
| Market_Context_Encoder | 0.1011 | 3.1x | **Too weak** |
| **Price_Patterns_Encoder** | **0.0324** | **1.0x** | **CRITICALLY LOW** |

### Why This Is Bad

1. **Price Patterns Ignored**:
   - CNN processing 8 features × 288 timesteps
   - Contains: candle structure, volume, multi-timeframe returns
   - Should be the MOST important feature group
   - But model barely uses it (0.032 activation)

2. **Session Timing Overused**:
   - Just 3 binary flags (Asia/London/NY session)
   - Simplest possible features
   - Model taking "shortcut": trade during certain hours
   - Not learning actual price patterns

3. **Market Context Underutilized**:
   - EMA/VWAP distances with tanh normalization
   - Critical for trend identification
   - Should guide directional bias
   - But overshadowed by session timing

### Root Causes

1. **Architectural Mismatch**:
   - Price CNN: 8 input → 32 output (only 4x expansion)
   - Session Linear: 3 input → 4 output (1.3x expansion)
   - Market MLP: 6 input → 16 output (2.7x expansion)
   - **Session encoder has disproportionate capacity per input feature**

2. **Network Takes Lazy Path**:
   - Learning from 3 binary session flags is trivial
   - Learning from 8×288 temporal price patterns is hard
   - Without proper capacity, network shortcuts to easier signals

3. **Gradient Flow Issues**:
   - CNN requires backward pass through conv layers + pooling
   - Session Linear is just one matrix multiply
   - Easier gradient flow → faster learning → dominates training

## Solution: Rebalanced Architecture

### Changes Made

| Component | Old | New | Reasoning |
|-----------|-----|-----|-----------|
| **Price Patterns CNN** | 32 dims | **64 dims** | 2x capacity - give price action proper representation |
| **Market Context MLP** | 16 dims | **32 dims** | 2x capacity - trend positioning is critical |
| **Trading Sessions** | 4 dims | **3 dims** | Reduce to minimal - discourage over-reliance |
| **Trend Indicators** | 32 dims | **32 dims** | No change - already balanced |
| **Account State** | 8 dims | **8 dims** | No change - appropriate for 5 features |
| **Position Info** | 8 dims | **8 dims** | No change - appropriate for 7 features |
| **Total Embedding** | 100 dims | **147 dims** | 47% increase in representation power |

### New Architecture

```
INPUT: 39 features per timestep
├── Price Patterns [8×288]  ──┐
│   ├── Conv1d(8→32, k=3)     │
│   ├── ReLU                  │  
│   ├── Conv1d(32→64, k=3)    │  ← DOUBLED capacity
│   ├── ReLU                  │
│   └── AdaptiveAvgPool1d     │
│   → Output: [64]  ──────────┼──┐
│                             │  │
├── Market Context [6×1]  ────┤  │
│   ├── Linear(6→32)          │  │  ← DOUBLED capacity
│   ├── ReLU                  │  │
│   └── Linear(32→32)         │  │
│   → Output: [32]  ──────────┼──┤
│                             │  │
├── Trend Indicators [10×1] ──┤  │
│   ├── Linear(10→32)         │  │
│   ├── ReLU                  │  │
│   └── Linear(32→32)         │  │
│   → Output: [32]  ──────────┼──┤
│                             │  │
├── Trading Sessions [3×1] ───┤  │
│   └── Linear(3→3)           │  │  ← REDUCED capacity
│   → Output: [3]  ───────────┼──┤
│                             │  │
├── Account State [5×1]  ─────┤  │
│   ├── Linear(5→16)          │  │
│   ├── ReLU                  │  │
│   └── Linear(16→8)          │  │
│   → Output: [8]  ───────────┼──┤
│                             │  │
├── Position Info [7×1]  ─────┤  │
│   ├── Linear(7→16)          │  │
│   ├── ReLU                  │  │
│   └── Linear(16→8)          │  │
│   → Output: [8]  ───────────┼──┤
│                             │  │
└── CONCATENATE  ─────────────┴──┘
    → Combined: [147]
    ├── Linear(147→128)
    ├── ReLU
    └── Linear(128→256)
    → Final Output: [256]
```

### Expected Improvements

After retraining with the rebalanced architecture:

**Target Feature Activations:**
```
Price_Patterns_Encoder:    0.15 - 0.25  (↑ 5-8x from 0.032)
Market_Context_Encoder:    0.15 - 0.20  (↑ 1.5-2x from 0.101)
Trend_Encoder:             0.10 - 0.15  (↓ slightly from 0.124)
Session_Encoder:           0.05 - 0.10  (↓ 4x from 0.399)
Account_Encoder:           0.10 - 0.15  (↓ slightly from 0.157)
Position_Encoder:          0.10 - 0.15  (↓ slightly from 0.140)
```

**Trading Behavior:**
- More responsive to actual price patterns (breakouts, reversals, patterns)
- Less dependent on "trade only during London session" heuristics
- Better adaptation to different market conditions
- Directional bias driven by EMA/VWAP positioning, not time-of-day

**Performance Metrics:**
- Better generalization to unseen data periods
- More balanced LONG/SHORT usage
- Reduced overfitting to training session patterns
- Higher win rate from better entry timing

## Implementation Details

### Files Modified

1. **src/environments/trading_enhanced_extractor.py**:
   - Updated CNN layers: Conv1d(32→32) → Conv1d(32→64)
   - Updated market context: Linear(6→16) → Linear(6→32) + Linear(32→32)
   - Reduced session: Linear(3→4) → Linear(3→3)
   - Updated fusion: combined_dim 100 → 147
   - Updated forward pass dimensions

### Combined with Reward Function v24

This architecture rebalancing works synergistically with the reward function fixes:

| Component | Issue | Fix |
|-----------|-------|-----|
| **Feature Extraction** | Over-relied on sessions | Rebalanced encoder capacities |
| **Reward Function** | Too risk-averse (96% HOLD) | Increased exploration bonus, reduced SL penalty |
| **Combined Effect** | Model shortcuts + won't trade | Price-driven trading with proper risk-taking |

## Testing Plan

### 1. Retrain Model
```python
# Use same training configuration
PRESET = 1  # 65k steps
# Model will automatically use new extractor architecture
```

### 2. Validate Feature Usage
```bash
python test_action_distribution.py
# Should show:
# - Price patterns activation: 0.15+
# - Session activation: <0.10
# - More balanced overall
```

### 3. Check Trading Behavior
```python
# Run RL_Tests.ipynb
# Expected:
# - HOLD: 40-60% (down from 96%)
# - LONG + SHORT: 30-50% (up from 4%)
# - Better entry timing (price-driven)
```

### 4. Compare Performance
```python
# Metrics to track:
# - Win rate
# - Average PnL per trade
# - Feature activation balance
# - Directional distribution (LONG vs SHORT)
```

## Risk Mitigation

### Potential Issues

1. **Training Instability**:
   - Larger network (147 vs 100 dims) = more parameters
   - May need slightly lower learning rate
   - **Mitigation**: Monitor loss/entropy in first few iterations

2. **Overfitting**:
   - More capacity could overfit to training data
   - **Mitigation**: Test on multiple unseen periods, track validation performance

3. **Convergence Time**:
   - Network might need more steps to learn patterns
   - **Mitigation**: Consider Preset 2 (131k steps) if needed

### Rollback Plan

If rebalancing causes problems:
```bash
# Restore old extractor
git checkout src/environments/trading_enhanced_extractor.py

# Or manually revert dimensions:
# Price: 64 → 32
# Market: 32 → 16  
# Session: 3 → 4
# Combined: 147 → 100
```

## Success Criteria

After retraining, the model is successful if:

1. ✅ Price_Patterns_Encoder activation ≥ 0.15
2. ✅ Session_Encoder activation ≤ 0.10
3. ✅ Feature activations more balanced (std < 0.05)
4. ✅ HOLD action < 70%
5. ✅ Win rate maintained or improved
6. ✅ Works on multiple test periods (not just one)

## Next Steps

1. ⏳ Retrain model with rebalanced extractor
2. ⏳ Run feature activation analysis
3. ⏳ Test on multiple unseen periods
4. ⏳ Compare vs baseline (old architecture)
5. ⏳ Fine-tune if needed (learning rate, epochs)

---

**Note**: This rebalancing should be done TOGETHER with Reward Function v24. Both fixes address different aspects of the same problem:
- Architecture: Ensures model CAN learn from price patterns
- Rewards: Ensures model WANTS to trade based on those patterns
