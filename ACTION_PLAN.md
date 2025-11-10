# Action Plan - Fix Trading Bot

## Current Status
- ❌ Old model performance: 12.5% win rate, -$467 loss, 100% SHORT
- ✅ Reward function updated (v23 → v24)
- ✅ Feature extractor rebalanced (100 → 147 dims)
- ⏳ **Retraining required** to apply changes

## Step-by-Step Plan

### 1. Retrain Model (Required)
```python
# Open RL_BACK.ipynb
# Run all cells to train with:
# - Reward Function v24 (exploration +10, idle penalty)
# - Extractor v2 (Price 64, Market 32, Session 3)
# 
# Expected training time: ~12-15 minutes (Preset 1)
```

**Monitor during training:**
- [ ] Entropy should stay around -1.0 to -2.0 (action diversity)
- [ ] Explained variance > 0.4 (model learning)
- [ ] Loss decreasing steadily
- [ ] No NaN values

### 2. Test Action Distribution
```bash
source .venv/Scripts/activate
python test_action_distribution.py
```

**Expected results:**
- [ ] HOLD: 40-60% (down from 96%)
- [ ] LONG: 15-25%
- [ ] SHORT: 15-25%
- [ ] CLOSE: 5-15%
- [ ] LONG/SHORT ratio balanced (0.5-2.0)

### 3. Test Feature Activations
```python
# Open RL_Tests.ipynb
# Run first cell to get feature activations
```

**Expected results:**
- [ ] Price_Patterns_Encoder: 0.15-0.25 (up from 0.032)
- [ ] Market_Context_Encoder: 0.15-0.20 (up from 0.054)
- [ ] Session_Encoder: 0.05-0.10 (down from 0.282)
- [ ] All encoders within 3x range (not 12x like before)

### 4. Evaluate Trading Performance
```python
# Continue running RL_Tests.ipynb cells
```

**Target metrics:**
- [ ] Win rate: > 40%
- [ ] Avg PnL/trade: > 0
- [ ] TP rate: > 30%
- [ ] SL rate: < 50%
- [ ] Direction balance: Both LONG and SHORT trades
- [ ] Total PnL: Positive over 1000+ steps

### 5. Test on Multiple Periods
```python
# In RL_Tests.ipynb, change test_start:
test_start = 10_000  # Try different periods
test_start = 50_000
test_start = 100_000
```

**Verify:**
- [ ] Model works in different market conditions
- [ ] Not just profitable in one specific period
- [ ] Direction balance maintained across periods

## Troubleshooting

### If Model Still Won't Trade (HOLD > 80%)
- Increase exploration bonus: 10.0 → 15.0
- Increase idle penalty: -0.5 → -1.0
- Check training entropy (should be > -2.0)

### If Still Directional Bias (>80% one direction)
- Check training data period for bias
- Verify feature engineering (EMA distances symmetric?)
- Add directional balance reward in environment

### If Win Rate Still Low (<30%)
- SL too tight (check ATR multipliers)
- TP too far (check RR ratios)
- Entry timing poor (need better features)
- Consider adding pattern recognition features

### If Feature Imbalance Persists
- Increase price encoder: 64 → 96 dims
- Reduce session encoder: 3 → 2 dims  
- Add dropout to session encoder (force network to use other features)

## Success Criteria

Model is ready for live paper trading when:
- ✅ Win rate > 45%
- ✅ Profit factor > 1.5 (gross profit / gross loss)
- ✅ Both LONG and SHORT used (20-80% each)
- ✅ Works on 3+ different test periods
- ✅ Feature activations balanced (max/min < 5x)
- ✅ Average PnL/trade > $10
- ✅ Max drawdown < 20%

## Quick Commands

```bash
# Activate environment
source .venv/Scripts/activate

# Train model
# (Open RL_BACK.ipynb in Jupyter/VSCode and run)

# Test action distribution
python test_action_distribution.py

# Test on different period
# (Edit test_start in RL_Tests.ipynb)

# Compare features
grep "Price_Patterns_Encoder" *.md
```

## Files Changed

✅ `src/environments/simple_trading_env.py` - Reward v24
✅ `src/environments/trading_enhanced_extractor.py` - Extractor v2
⏳ `trading_bot.zip` - Needs retraining
⏳ Model performance - Will improve after retraining

## Timeline

- Retraining: 12-15 minutes
- Testing: 5 minutes
- Validation: 10 minutes
- Total: ~30 minutes

**Next immediate action:** Run RL_BACK.ipynb to retrain the model!
