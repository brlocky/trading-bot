# 🚀 TODO: Major Refactor Plan

## Phase 0: Feature Optimization (PREREQUISITE - IN PROGRESS)

### Step 0.1: Identify & Remove Redundant Features
**Why:** CNNs already learn these patterns from raw price data

 
## Phase 2: Add Oscillator CNN for Divergences
- Create `OscillatorCNN` module to detect divergence patterns.
  - Input: 3 channels (RSI, MACD, MACD Signal).
  - Architecture: 3 Conv1D layers → Multi-scale divergence detectors → Fully connected.
- Add to `TradingCombinedExtractor`.
- Train and verify that the model learns to detect divergences.

## Phase 3: Add Trendline Detection

### Option A: Classical Peak Detection (Faster)
- Use `scipy.signal.find_peaks` to detect highs/lows.
- Fit trendlines using RANSAC.
- Add 6 trendline features to observation space:
  - Support slope, support distance, resistance slope, resistance distance.
  - Binary flags: near support, near resistance.

### Option B: Learnable Trendline CNN (More Advanced)
- Implement `TrendlineAttentionCNN`:
  - Detect pivots using Conv1D.
  - Use attention to focus on significant pivots.
  - Learn to connect pivots into trendlines.
- Add trendline features (slope, distance, breaks) to observation space.

## Phase 4: Retrain & Validate
- Train with the new architecture (no VecNormalize, Oscillator CNN, trendlines).
- Verify:
  - Divergence detection works (RSI/MACD).
  - Trendline breaks trigger trades.
  - Improved generalization to unseen data.

---

## Phase 5: Add High-Signal Features (AFTER Phase 0-4 validates)

### Step 5.1: Implement Divergence Detection
**Priority: CRITICAL** - Divergences predict reversals with 70%+ accuracy

**Create:** `src/features/divergence_detector.py`
```python
def detect_divergences(price, rsi, macd, lookback=48):
    # Detects bullish/bearish divergences
    # Returns: [bullish_div, bearish_div, strength]
```

**New Features:**
- `rsi_price_divergence_1h` - RSI makes lower low, price makes higher low
- `rsi_price_divergence_4h` - Multi-timeframe confirmation
- `macd_price_divergence_1h` - MACD divergence
- `macd_histogram_divergence` - Hidden divergence

### Step 5.2: Add Order Flow Features
**Priority: HIGH** - Shows smart money activity

**Enhance:** `src/features/enhanced_volume_profile.py`
- `delta_volume` - Buy volume - sell volume
- `cvd` - Cumulative Volume Delta
- `absorption` - Large orders absorbing moves
- `iceberg_detection` - Hidden large orders

### Step 5.3: Add Multi-Timeframe Context
**Priority: MEDIUM** - Higher timeframes = stronger signals

**New Features:**
- `rsi_15m`, `rsi_1h`, `rsi_4h` - MTF RSI
- `trend_15m`, `trend_1h`, `trend_4h` - HTF bias
- `vp_poc_15m`, `vp_poc_1h` - Key levels across timeframes

### Step 5.4: Add Market Regime Features
**Priority: MEDIUM** - Different strategies for different regimes

**New Features:**
- `volatility_regime` - Low/Normal/High
- `liquidity_regime` - Volume clusters from VP
- `market_phase` - Accumulation/Markup/Distribution/Markdown

---

## Phase 6: Advanced Architecture (FUTURE)

### Step 6.1: Remove VecNormalize
- Replace VecNormalize with manual normalization in `SimpleTradingEnv`
- Compute normalization stats (mean, std) from training data
- Normalize each feature group separately
- Save normalization parameters with the model

### Step 6.2: Add Oscillator CNN for Divergences
- Create `OscillatorCNN` module
  - Input: 3 channels (RSI, MACD, MACD Signal)
  - Architecture: Conv1D → Multi-scale divergence detectors
- Add to `TradingCombinedExtractor`
- Let CNN learn divergence patterns automatically

### Step 6.3: Add Trendline Detection

**Option A: Classical Peak Detection (Faster)**
- Use `scipy.signal.find_peaks` for highs/lows
- Fit trendlines using RANSAC
- Add 6 trendline features

**Option B: Learnable Trendline CNN (Better)**
- Implement `TrendlineAttentionCNN`
- Use attention to focus on significant pivots
- Learn to connect pivots into trendlines

---

## 🎯 Expected Benefits

**After Phase 0 (Remove Redundancy):**
- ✅ Less overfitting (fewer correlated features)
- ✅ Faster training (smaller observation space)
- ✅ Better generalization (high-quality signals only)
- ✅ Clearer insights (interpretable features)

**After Phase 2 (Add High-Signal Features):**
- ✅ Divergence-based entries (70%+ accuracy)
- ✅ Smart money tracking (order flow)
- ✅ Multi-timeframe confirmation (stronger signals)
- ✅ Regime-aware trading (adapt to conditions)

**After Phase 0 (Remove Redundancy):**
- ✅ Less overfitting (fewer correlated features)
- ✅ Faster training (smaller observation space)
- ✅ Better generalization (high-quality signals only)
- ✅ Clearer insights (interpretable features)

**After Phase 1-4 (Core Architecture):**
- ✅ More robust (no VecNormalize hiding shifts)
- ✅ Automatic divergence detection (Oscillator CNN)
- ✅ Trendline awareness (support/resistance breaks)
- ✅ Better interpretability (visualize patterns)

**After Phase 5 (Add High-Signal Features):**
- ✅ Divergence-based entries (70%+ accuracy)
- ✅ Smart money tracking (order flow)
- ✅ Multi-timeframe confirmation (stronger signals)
- ✅ Regime-aware trading (adapt to conditions)

**After Phase 6 (Advanced Architecture):**
- ✅ Automatic divergence detection (Oscillator CNN)
- ✅ Trendline awareness (support/resistance breaks)
- ✅ More robust (no VecNormalize hiding shifts)
- ✅ Better interpretability (visualize patterns)

---

## 📋 Implementation Order

1. ✅ **Phase 0.1**: Identify redundant indicators (DONE)
2. ⏳ **Phase 0.2**: Update `indicator_utils.py`
3. ⏳ **Phase 0.3**: Train & validate lean feature set
4. ⏳ **Phase 1**: Remove VecNormalize (manual normalization)
5. ⏳ **Phase 2**: Add Oscillator CNN for divergences
6. ⏳ **Phase 3**: Add Trendline Detection
7. ⏳ **Phase 4**: Retrain & validate full architecture
8. ⏳ **Phase 5.1-5.4**: Add high-signal features (divergence, order flow, MTF, regime)
9. ⏳ **Phase 6**: Advanced architecture optimizations

**Current Status:** Waiting for v10 training to complete, then will start Phase 0.2
- **Trendline awareness**: Trades support/resistance breaks.
- **More robust**: No VecNormalize hiding distribution shifts.
- **Cleaner code**: Explicit normalization.
- **Better interpretability**: Can visualize trendlines and divergences.