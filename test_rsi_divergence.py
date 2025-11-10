"""
Test RSI Divergence Feature Normalization

Verifies that RSI + High + Low are properly normalized to [-1, 1] range.
"""

import pandas as pd
import numpy as np
from src.data_processing.enhanced_features import precompute_rsi_divergence_features
from src.utils.indicator_utils import add_indicators

# Load sample data
print("Loading data...")
df = pd.read_json('data/BTCUSDT-15m.json')
df['date'] = pd.to_datetime(df['date_close'])
add_indicators(df)
df = df.dropna().reset_index(drop=True)

print(f"\nDataframe shape: {df.shape}")
print(f"Columns available: {list(df.columns)}")

# Compute RSI divergence features
print("\nComputing RSI divergence features...")
feature_cols = precompute_rsi_divergence_features(df, window=288)

print(f"\nFeature columns created: {feature_cols}")
print(f"Expected: ['rsi_divergence', 'high_divergence', 'low_divergence']")

# Check normalization ranges
print("\n" + "="*80)
print("NORMALIZATION VALIDATION")
print("="*80)

for col in feature_cols:
    values = df[col].values
    print(f"\n{col}:")
    print(f"  Min:    {values.min():.6f}")
    print(f"  Max:    {values.max():.6f}")
    print(f"  Mean:   {values.mean():.6f}")
    print(f"  Median: {np.median(values):.6f}")
    print(f"  Std:    {values.std():.6f}")

    # Check if in [-1, 1] range
    in_range = (values >= -1.0 - 1e-6) & (values <= 1.0 + 1e-6)
    print(f"  In [-1, 1]: {in_range.sum()}/{len(values)} ({100*in_range.sum()/len(values):.2f}%)")

    if not in_range.all():
        out_of_range = values[~in_range]
        print(f"  ⚠️  OUT OF RANGE VALUES: {len(out_of_range)}")
        print(f"     Min out: {out_of_range.min():.6f}")
        print(f"     Max out: {out_of_range.max():.6f}")

# Test on sample window
print("\n" + "="*80)
print("SAMPLE WINDOW TEST (last 10 candles)")
print("="*80)

sample = df[feature_cols].tail(10)
print(sample.to_string())

# Test divergence detection potential
print("\n" + "="*80)
print("DIVERGENCE PATTERN CHECK")
print("="*80)

# Look for potential bullish divergence: price falling, RSI rising
window = 50
for i in range(len(df) - window, len(df) - 10):
    rsi_change = df.loc[i+window-1, 'rsi_divergence'] - df.loc[i, 'rsi_divergence']
    low_change = df.loc[i+window-1, 'low_divergence'] - df.loc[i, 'low_divergence']

    # Bullish divergence: RSI up, Low down
    if rsi_change > 0.2 and low_change < -0.2:
        print(f"\n✓ Potential BULLISH divergence at index {i}")
        print(f"  RSI change:  {rsi_change:+.4f} (rising)")
        print(f"  Low change:  {low_change:+.4f} (falling)")
        print(f"  Date: {df.loc[i+window-1, 'date']}")

    # Bearish divergence: RSI down, High up
    if rsi_change < -0.2 and (df.loc[i+window-1, 'high_divergence'] - df.loc[i, 'high_divergence']) > 0.2:
        high_change = df.loc[i+window-1, 'high_divergence'] - df.loc[i, 'high_divergence']
        print(f"\n✓ Potential BEARISH divergence at index {i}")
        print(f"  RSI change:  {rsi_change:+.4f} (falling)")
        print(f"  High change: {high_change:+.4f} (rising)")
        print(f"  Date: {df.loc[i+window-1, 'date']}")

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
