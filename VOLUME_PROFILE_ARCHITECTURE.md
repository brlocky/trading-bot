# Volume Profile Architecture - Complete Technical Explanation

## Data Structure Overview

The Volume Profile system has **two separate but related** data structures:

### 1. Price Bins (`vp.bins`)
- **Type**: Torch tensor of PRICE values
- **Shape**: (n_bins + 1,) = (51,) for 50 bins
- **Content**: Price bin edges from `linspace(price_min, price_max, n_bins+1)`
- **Example**: [99.0, 99.7, 100.4, 101.1, ..., 106.0]
- **Purpose**: Define price ranges for bucketing volume data
- **Updated**: Every bar, recalculated based on current session high/low

### 2. Volume Weights (`vp.weights`)
- **Type**: Torch tensor of VOLUME distribution
- **Shape**: (n_bins,) = (50,) for 50 bins
- **Content**: Normalized volume weights (sum = 1.0)
- **Example**: [0.0, 0.1, 0.1, 0.1, 0.0, 0.2, 0.2, ...]
- **Purpose**: Show how volume is distributed across price levels
- **Updated**: Every bar, accumulated using `index_add_`

## Data Flow

```
Input: OHLCV bar
    ↓
Split into High/Low with volume weight 0.5 each
    ↓
Map prices to bin indices using torch.bucketize(prices, bins)
    ↓
Accumulate volume: weights.index_add_(0, bin_indices, volumes)
    ↓
Normalize: weights /= weights.sum()
    ↓
Store snapshot: daily_bins_history[idx] = weights.clone()
```

## What Each Function Returns

### `vp_obj.bins`
```python
# PRICE bin edges (n_bins + 1 edges define n_bins ranges)
tensor([99.0, 99.7, 100.4, ..., 106.0])  # Shape: (51,)
# Edge 0-1 = bin 0, Edge 1-2 = bin 1, etc.
```

### `vp_obj.weights`
```python
# VOLUME distribution (current session, normalized)
tensor([0.0, 0.1, 0.1, ..., 0.2])  # Shape: (50,), Sum = 1.0
# weights[i] = fraction of total volume in price range [bins[i], bins[i+1]]
```

### `vp_obj.get_bins_history(lookback)`
```python
# VOLUME weight snapshots over time
tensor([[...],  # t-lookback: volume distribution
        [...],  # t-lookback+1
        [...],  # ...
        [...]])  # t (most recent)
# Shape: (lookback, n_bins) = (288, 50)
# Each row is a snapshot of vp.weights from that timestep
```

## Feature Extraction Logic

### Group 9: `get_volume_profile_bins()`

**Returns**: (lookback, 56) tensor

**Channels**:
- **0-49**: Volume distribution from `get_bins_history()` - shows WHERE volume is concentrated
- **50**: VAH marker - normalized position [0, 1] where 0=bottom bin, 1=top bin
- **51**: POC marker - normalized position [0, 1]
- **52**: VAL marker - normalized position [0, 1]
- **53**: Close price marker - normalized position [0, 1] within current range
- **54**: High price marker - normalized position [0, 1]
- **55**: Low price marker - normalized position [0, 1]

**Marker Encoding Example**:
```python
# Current price range: bins = [99.0, 99.7, 100.4, ..., 106.0]
# VAH = 103.5
# Map to bin: (103.5 - 99.0) / (106.0 - 99.0) * 49 = 31.5 → bin 31
# Normalize: 31 / 49 = 0.633
# Result: markers[:, 0] = 0.633
# 
# CNN interpretation: "VAH is at 63.3% up the distribution"
```

### Group 10: `get_previous_day_volume_profile_bins()`

**Returns**: (lookback, 60) tensor

**Channels**:
- **0-49**: Yesterday's volume distribution (static, same for all timesteps)
- **50**: Yesterday's High - normalized position in yesterday's range
- **51**: Yesterday's VAH - normalized position
- **52**: Yesterday's POC - normalized position
- **53**: Yesterday's VAL - normalized position
- **54**: Yesterday's Low - normalized position
- **55**: Today's Close position - how today's close relates to yesterday's range
- **56**: Today's High position - within yesterday's range
- **57**: Today's Low position - within yesterday's range
- **58**: 2-day-ago VAH - for multi-day context
- **59**: 2-day-ago VAL - for multi-day context

**Cross-Day Mapping Example**:
```python
# Yesterday's range: prev_bins = [100.0, 100.5, ..., 110.0]
# Today's close = 105.5
# Map to yesterday's bins: (105.5 - 100.0) / (110.0 - 100.0) * 49 = 26.95 → bin 27
# Normalize: 27 / 49 = 0.551
# Result: markers[t, 5] = 0.551
#
# CNN interpretation: "Today's close is at 55.1% of yesterday's range"
```

## Why Normalized Position Encoding [0, 1]?

### Previous (WRONG) Implementation:
```python
markers[:, 0] = 1.0  # VAH exists
markers[:, 1] = 1.0  # POC exists
markers[:, 2] = 1.0  # VAL exists
```
**Problem**: CNN knows "there is a VAH" but NOT "where the VAH is"

### Current (CORRECT) Implementation:
```python
markers[:, 0] = float(vah_bin) / (n_bins - 1)  # VAH at position 0.633
markers[:, 1] = float(poc_bin) / (n_bins - 1)  # POC at position 0.500
markers[:, 2] = float(val_bin) / (n_bins - 1)  # VAL at position 0.367
```
**Benefit**: CNN learns "VAH is at 63% up the distribution, POC at 50%, VAL at 37%"

This encoding is:
- **Scale-invariant**: Works for BTC at $100K or $1K
- **Spatially meaningful**: Higher values = higher in the distribution
- **Natural for CNNs**: Continuous [0, 1] values are easy to learn
- **Relative positioning**: Shows relationships between levels

## CNN Processing

The Conv1d layers process these features to learn patterns like:

1. **Volume Shelf Detection**: High volume at a specific level
   - Pattern: Spike in channels 0-49 at position X, with marker at X
   
2. **Price Rejection**: Price bounces off VAH/VAL
   - Pattern: Close marker approaches level marker, then reverses
   
3. **Gap Fills**: Price returns to POC after moving away
   - Pattern: Close marker distance from POC marker changes over time

4. **Multi-Day Context**: Today's price vs yesterday's structure
   - Pattern: Today's markers (55-57) relative to yesterday's markers (50-54)

## Verification

Run `test_bins_understanding.py` to see actual data:
```bash
.venv/Scripts/python.exe test_bins_understanding.py
```

Output shows:
- `vp.bins` = Price edges [99.0, 99.7, ..., 106.0]
- `vp.weights` = Volume distribution [0.0, 0.1, ..., 0.2] (sum=1.0)
- `get_bins_history()` = Volume weight snapshots over time

## Summary

✅ **`vp.bins`** = PRICE bin edges (for mapping prices to indices)  
✅ **`vp.weights`** = VOLUME distribution (what the CNN actually learns from)  
✅ **Marker encoding** = Normalized [0, 1] positions (spatial information for CNN)  
✅ **Implementation** = CORRECT - properly maps prices to bins, then normalizes

The architecture cleanly separates:
1. **Price bucketing** (bins edges)
2. **Volume distribution** (weights)
3. **Spatial encoding** (normalized marker positions)

This allows the CNN to learn both volume patterns AND spatial price relationships.
