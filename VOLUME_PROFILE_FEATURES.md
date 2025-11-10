# Volume Profile Features Documentation

## Overview
The `EnhancedVolumeProfile` outputs **26 features** per timestep, plus **54 VP distribution bins** for CNN processing.

## ⚠️ ACTUAL IMPLEMENTATION
- **Volume Profile Features**: 26 features (Group 8)
- **VP Distribution Bins**: 54 features (Group 9) - 50 bins + 4 markers (VAH/VAL/POC/Close)
- **Total VP-related features**: 80 per timestep

---

## Feature Breakdown

### Group 8: Volume Profile Core Features (26 features)

#### CORE DISTANCES (Features 0-9)

| Index | Feature Name | Description | Calculation | Range |
|-------|-------------|-------------|-------------|-------|
| 0 | Distance to CURRENT day VAH | Distance from price to today's Value Area High | `(price - current_vah) / price` | Typically ±0.02 |
| 1 | Distance to CURRENT day VAL | Distance from price to today's Value Area Low | `(price - current_val) / price` | Typically ±0.02 |
| 2 | Distance to CURRENT day POC | Distance from price to today's Point of Control | `(price - current_poc) / price` | Typically ±0.01 |
| 3 | Distance to CURRENT day High | Distance to today's high (intraday level) | `(price - current_day_high) / price` | Typically ±0.01 |
| 4 | Distance to CURRENT day Low | Distance to today's low (intraday level) | `(price - current_day_low) / price` | Typically ±0.01 |
| 5 | Distance to PREVIOUS day VAH | Distance from price to yesterday's Value Area High | `(price - prev_vah) / price` | Typically ±0.03 |
| 6 | Distance to PREVIOUS day VAL | Distance from price to yesterday's Value Area Low | `(price - prev_val) / price` | Typically ±0.03 |
| 7 | Distance to PREVIOUS day POC | Distance from price to yesterday's Point of Control | `(price - prev_poc) / price` | Typically ±0.02 |
| 8 | Distance to PREVIOUS day High | Distance to yesterday's high | `(price - prev_day_high) / price` | Typically ±0.02 |
| 9 | Distance to PREVIOUS day Low | Distance to yesterday's low | `(price - prev_day_low) / price` | Typically ±0.02 |

#### POSITION FLAGS (Features 10-17)

| Index | Feature Name | Description | Values |
|-------|-------------|-------------|--------|
| 10 | CURRENT day: Above VAH | Price is above today's Value Area High | 1.0 or 0.0 |
| 11 | CURRENT day: Between VAH-POC | Price is between today's VAH and POC | 1.0 or 0.0 |
| 12 | CURRENT day: Between POC-VAL | Price is between today's POC and VAL | 1.0 or 0.0 |
| 13 | CURRENT day: Below VAL | Price is below today's Value Area Low | 1.0 or 0.0 |
| 14 | PREVIOUS day: Above VAH | Price is above yesterday's Value Area High | 1.0 or 0.0 |
| 15 | PREVIOUS day: Between VAH-POC | Price is between yesterday's VAH and POC | 1.0 or 0.0 |
| 16 | PREVIOUS day: Between POC-VAL | Price is between yesterday's POC and VAL | 1.0 or 0.0 |
| 17 | PREVIOUS day: Below VAL | Price is below yesterday's Value Area Low | 1.0 or 0.0 |

#### NAKED POC FEATURES (Features 18-21)

| Index | Feature Name | Description | Normalization | Range |
|-------|-------------|-------------|---------------|-------|
| 18 | Naked Daily POCs Count | Number of untouched daily POCs | `count / max_naked_pocs` (max=10) | 0.0 to 1.0 |
| 19 | Naked Weekly POCs Count | Number of untouched weekly POCs | `count / max_naked_pocs` (max=10) | 0.0 to 1.0 |
| 20 | Nearest Naked Daily POC Distance | Distance to closest untouched daily POC | `distance / price` | Typically ±0.05 |
| 21 | Nearest Naked Weekly POC Distance | Distance to closest untouched weekly POC | `distance / price` | Typically ±0.10 |

#### NAKED VAH/VAL FEATURES (Features 22-25)

| Index | Feature Name | Description | Normalization | Range |
|-------|-------------|-------------|---------------|-------|
| 22 | Naked Daily VAHs Count | Number of untouched daily Value Area Highs | `count / max_naked_pocs` (max=10) | 0.0 to 1.0 |
| 23 | Naked Daily VALs Count | Number of untouched daily Value Area Lows | `count / max_naked_pocs` (max=10) | 0.0 to 1.0 |
| 24 | Nearest Naked Daily VAH Distance | Distance to closest untouched daily VAH | `distance / price` | Typically ±0.05 |
| 25 | Nearest Naked Daily VAL Distance | Distance to closest untouched daily VAL | `distance / price` | Typically ±0.05 |

---

### Group 9: VP Distribution (54 features - for CNN processing)

Processed by 1D CNN in `TradingCombinedExtractor` to detect volume accumulation patterns and spatial relationships.

| Component | Count | Description | Range |
|-----------|-------|-------------|-------|
| **Volume Bins** | 50 | Normalized volume distribution across price levels | 0.0 to 1.0 |
| **VAH Marker** | 1 | Binary marker indicating Value Area High position | 1.0 at VAH bin, 0.0 elsewhere |
| **VAL Marker** | 1 | Binary marker indicating Value Area Low position | 1.0 at VAL bin, 0.0 elsewhere |
| **POC Marker** | 1 | Binary marker indicating Point of Control position | 1.0 at POC bin, 0.0 elsewhere |
| **Close Marker** | 1 | Binary marker indicating current close price position | 1.0 at close bin, 0.0 elsewhere |
| **TOTAL** | **54** | Used by `vp_bins_cnn` for spatial pattern detection | - |

**What the CNN learns:**
- Volume accumulation zones (high volume nodes)
- Price rejection areas (low volume zones)
- Spatial relationship between price and key levels
- Dynamic support/resistance based on volume
- Institutional order flow patterns

---

## Usage in Trading Environment

### Environment Configuration
```python
# From simple_trading_env.py
self.observation_space = gym.spaces.Dict({
    # ... other groups ...
    'volume_profile': gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(lookback_window, 26),  # 26 VP features
        dtype=np.float32
    ),
    'vp_distribution': gym.spaces.Box(
        low=0, high=1,
        shape=(lookback_window, 54),  # 50 bins + 4 markers
        dtype=np.float32
    ),
})
```

### Neural Network Processing
From `TradingCombinedExtractor`:

```python
# 1. Volume Profile Features (26) → Transformer
self.vp_projection = nn.Linear(26, 48)  # Project 26 features to 48-dim
self.vp_transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, dropout=0.1),
    num_layers=2
)
self.vp_output = nn.Linear(48, 24)  # Output 24-dim embedding

# 2. VP Distribution Bins (54) → CNN
self.vp_bins_cnn = nn.Sequential(
    # Input: [batch, 54, 288] (54 channels, 288 timesteps)
    nn.Conv1d(54, 64, kernel_size=5, padding=2),  # Detect local volume patterns
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Conv1d(64, 64, kernel_size=3, padding=1),  # Refine patterns
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Conv1d(64, 32, kernel_size=3, padding=1),  # Compress features
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1)  # Global pooling → [batch, 32, 1]
)
self.vp_bins_output = nn.Linear(32, 16)  # Output 16-dim embedding

# Total VP contribution: 24 + 16 = 40 dimensions to final embedding
```

---

## Complete Environment Feature Count

Based on `SimpleTradingEnv` and `TradingCombinedExtractor`:

| # | Group Name | Features | Processing | Output Dim |
|---|------------|----------|------------|------------|
| 1 | Price OHLC Spatial | 4 | Conv2d (Candlestick patterns) | 32 |
| 2 | Price OHLC Temporal | 4 | Conv1d (Trend evolution) | 64 |
| 3 | RSI Divergence | 2 | Conv1d (RSI divergence) | 32 |
| 4 | MACD Divergence | 3 | Conv1d (MACD divergence) | 32 |
| 5 | Price Context | 12 | Transformer | 32 |
| 6 | Trend Indicators | 10 | Transformer | 32 |
| 7 | Momentum Oscillators | 2 | MLP | 24 |
| 8 | **Volume Profile** | **26** | **Transformer** | **24** |
| 9 | **VP Distribution** | **54** | **CNN** | **16** |
| 10 | Trading Sessions | 3 | MLP | 4 |
| 11 | Account State | 4 | MLP | 8 |
| 12 | Position Info | 7 | MLP | 8 |
| 13 | Performance Metrics | 7 | MLP | 8 |
| **TOTAL** | **138** | - | **Combined** | **316 → 256** |

### Feature Distribution:
- **Price/Technical**: 37 features (27%)
- **Volume Profile**: 80 features (58%) ← **Largest component!**
- **Trading/Account**: 21 features (15%)

---

## Example Feature Values

At 10:00 AM on a typical trading day:

```python
volume_profile_features = [
    # CORE DISTANCES (0-9): Current day levels
    -0.0015,  # 0: Distance to current VAH (15 bps below)
    0.0025,   # 1: Distance to current VAL (25 bps above)
    -0.0008,  # 2: Distance to current POC (8 bps below)
    -0.0012,  # 3: Distance to current day high (12 bps below)
    0.0035,   # 4: Distance to current day low (35 bps above)
    
    # CORE DISTANCES (5-9): Previous day levels
    -0.0020,  # 5: Distance to prev VAH (20 bps below)
    0.0030,   # 6: Distance to prev VAL (30 bps above)
    -0.0010,  # 7: Distance to prev POC (10 bps below)
    -0.0018,  # 8: Distance to prev day high (18 bps below)
    0.0040,   # 9: Distance to prev day low (40 bps above)
    
    # POSITION FLAGS (10-17): Binary indicators
    0.0,      # 10: NOT above current VAH
    1.0,      # 11: Between current VAH-POC ✓
    0.0,      # 12: NOT between current POC-VAL
    0.0,      # 13: NOT below current VAL
    0.0,      # 14: NOT above prev VAH
    1.0,      # 15: Between prev VAH-POC ✓
    0.0,      # 16: NOT between prev POC-VAL
    0.0,      # 17: NOT below prev VAL
    
    # NAKED POC FEATURES (18-21)
    0.3,      # 18: 3 naked daily POCs (normalized)
    0.1,      # 19: 1 naked weekly POC (normalized)
    -0.0025,  # 20: Nearest daily POC 25 bps below
    0.0051,   # 21: Nearest weekly POC 51 bps above
    
    # NAKED VAH/VAL FEATURES (22-25)
    0.2,      # 22: 2 naked daily VAHs (normalized)
    0.3,      # 23: 3 naked daily VALs (normalized)
    -0.0030,  # 24: Nearest daily VAH 30 bps below
    0.0045,   # 25: Nearest daily VAL 45 bps above
]

# VP Distribution (54 features) - Example for one timestep
vp_distribution = [
    # Bins 0-49: Volume distribution (normalized 0-1)
    0.02, 0.03, 0.05, ..., 0.15, ..., 0.08, ..., 0.01,  # 50 bins
    
    # Markers (channels 50-53)
    0.0,   # 50: VAH marker (not at this bin)
    0.0,   # 51: VAL marker (not at this bin)
    1.0,   # 52: POC marker (at this bin!) ← Highest volume
    1.0,   # 53: Close marker (price at this bin) ← Current price
]
```

---

## Key Trading Insights

### What Makes Volume Profile Powerful

1. **Dual Time Perspective**
   - **Current Day (0-4, 10-13)**: Intraday context, real-time levels
   - **Previous Day (5-9, 14-17)**: Swing levels, overnight gaps

2. **Naked Levels as Magnets**
   - **Naked POCs (18-21)**: Price tends to revisit untouched POCs
   - **Naked VAH/VAL (22-25)**: Key support/resistance zones
   - Acts as "unfinished business" in the market

3. **Position Context**
   - **Above VAH**: Bullish breakout zone (features 10, 14)
   - **Between VAH-POC**: Neutral-bullish zone (features 11, 15)
   - **Between POC-VAL**: Neutral-bearish zone (features 12, 16)
   - **Below VAL**: Bearish breakdown zone (features 13, 17)

4. **CNN Pattern Recognition**
   - Volume accumulation = support/resistance
   - Low volume = price rejection zones
   - Spatial distance to POC = mean reversion signal
   - Marker positions = institutional levels

---

## Performance Optimizations

### ✅ Implemented
- **Typical Price**: Uses `(H+L+C)/3` instead of OHLC (4x faster)
- **Update Frequency**: VP updates every 3 bars (3x faster)
- **Bin Count**: Reduced from 100 to 50 bins (2x faster)
- **Combined speedup**: ~20-24x faster than original

### ✅ Removed Features
- **Cumulative VP Distribution** (54 features) - REMOVED in v0.2
  - Was taking 78% of `_get_obs()` time (~5ms per step)
  - Not critical for intraday trading decisions
  - Saved significant computational overhead

### 📊 Memory Usage
- **26 features × 288 timesteps × 4 bytes** = ~30 KB per observation
- **54 bins × 288 timesteps × 4 bytes** = ~62 KB per observation
- **Total VP memory**: ~92 KB per observation (manageable)

---

## Normalization Summary

| Feature Range | Normalization Method | Typical Values |
|---------------|---------------------|----------------|
| **Distances (0-9, 20-21, 24-25)** | Percentage of price | ±0.01 to ±0.05 (1-5%) |
| **Position Flags (10-17)** | Binary | 0.0 or 1.0 |
| **Counts (18-19, 22-23)** | Ratio of max (10) | 0.0 to 1.0 |
| **VP Bins (50 channels)** | Normalized volume | 0.0 to 1.0 |
| **Markers (4 channels)** | Binary | 0.0 or 1.0 |

---

## Changes from Previous Documentation

| Aspect | Old Doc | Actual Implementation | Status |
|--------|---------|----------------------|--------|
| VP Feature Count | 15 | **26** | ❌ Fixed |
| VP Distribution | Not mentioned | **54 features** | ❌ Fixed |
| Total Features | 78 | **138** | ❌ Fixed |
| Feature Details | Partial (0-14) | **Complete (0-25)** | ✅ Fixed |
| Cumulative VP | Not mentioned | Removed for performance | ✅ Documented |
| Processing | Not specified | Transformer + CNN | ✅ Documented |
| Output Embedding | Not specified | 24-dim + 16-dim | ✅ Documented |
| Memory Usage | Not mentioned | ~92 KB per obs | ✅ Documented |

---

## Trading Strategy Implications

### Mean Reversion Signals
- **Distance to POC** (Features 2, 7): Price tends to revert to POC
- **Naked POCs** (Features 18-21): Strong magnet effect
- **Position flags** (Features 10-17): Entry zones for mean reversion

### Breakout Signals
- **Above VAH** (Features 10, 14): Bullish breakout confirmation
- **Below VAL** (Features 13, 17): Bearish breakdown confirmation
- **Volume bins**: High volume = validated breakout

### Support/Resistance
- **Naked VAH/VAL** (Features 22-25): Key S/R levels
- **Previous day levels** (Features 5-9): Swing trade S/R
- **VP distribution**: Volume nodes = strong S/R

---

## Implementation Notes

### Minimum Data Requirement
- VP calculation requires **24 bars (2 hours)** of data
- Prevents invalid VP data from early session (first 5-10 minutes)
- Falls back to previous day's values when insufficient data

### Update Frequency
- VP updates every **3 bars** (not every bar)
- Balance between freshness and computational cost
- Sufficient for 5-minute timeframe trading

### Device Placement
- All VP tensors are created on the specified device (CPU/CUDA)
- No device transfers needed during observation generation
- Optimized for GPU training with large batch sizes