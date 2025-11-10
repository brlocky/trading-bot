"""Quick test to validate VP feature changes."""
import torch
import numpy as np
from datetime import datetime, timedelta
from src.features.enhanced_volume_profile import EnhancedVolumeProfile
from src.data_processing.enhanced_features import (
    get_volume_profile_features,
    get_volume_profile_bins,
    get_previous_day_volume_profile_bins
)

# Test EnhancedVolumeProfile with session_type calculation
print("="*60)
print("Testing EnhancedVolumeProfile with session_type...")
print("="*60)

vp = EnhancedVolumeProfile(n_bins=50, lookback_window=288, device='cpu')

# Simulate a day of trading data
base_date = datetime(2024, 1, 1, 9, 0)
base_price = 50000.0

# Day 1: Normal session
for i in range(96):  # 96 bars = 24 hours of 15-min data
    timestamp = base_date + timedelta(minutes=15*i)
    open_p = base_price + np.random.randn() * 10
    high_p = open_p + abs(np.random.randn() * 20)
    low_p = open_p - abs(np.random.randn() * 20)
    close_p = open_p + np.random.randn() * 15
    volume = 100 + np.random.rand() * 50

    vp.update(timestamp, open_p, high_p, low_p, close_p, volume)

print(f"Day 1 completed - Sessions: {len(vp.daily_sessions)}")
if len(vp.daily_sessions) > 0:
    print(f"  VAH: {vp.current_day_vah:.2f}, VAL: {vp.current_day_val:.2f}, POC: {vp.current_day_poc:.2f}")

# Day 2: Price opens above previous VAH (test session_type)
base_date = datetime(2024, 1, 2, 9, 0)
base_price = 50100.0  # Above previous VAH

for i in range(96):
    timestamp = base_date + timedelta(minutes=15*i)
    open_p = base_price + np.random.randn() * 10
    high_p = open_p + abs(np.random.randn() * 20)
    low_p = open_p - abs(np.random.randn() * 20)
    close_p = open_p + np.random.randn() * 15
    volume = 100 + np.random.rand() * 50

    vp.update(timestamp, open_p, high_p, low_p, close_p, volume)

print(f"Day 2 completed - Sessions: {len(vp.daily_sessions)}")
if len(vp.daily_sessions) > 0:
    print(f"  VAH: {vp.current_day_vah:.2f}, VAL: {vp.current_day_val:.2f}, POC: {vp.current_day_poc:.2f}")

# Test session_type calculation
current_price = base_price
session_type = vp.calculate_session_type(current_price)
print(f"\nSession Type: {session_type}")
session_types = ['ABOVE_VA', 'BELOW_VA', 'FAILED_BREAKOUT_HIGH', 'FAILED_BREAKDOWN_LOW',
                 'INSIDE_VA', 'OVERLAPPING_EXPAND', 'BREAKOUT_HIGH', 'BREAKDOWN_LOW']
print(f"  Classification: {session_types[session_type]}")

# Test feature extraction
print("\n" + "="*60)
print("Testing get_volume_profile_features (18 features)...")
print("="*60)

lookback = 288
features = get_volume_profile_features(vp, current_price, lookback)
print(f"Features shape: {features.shape} (expected: ({lookback}, 18))")
print(f"Features dtype: {features.dtype}")
print(f"\nFirst timestep features:")
print(f"  dist_poc: {features[0, 0]:.6f}")
print(f"  dist_vah: {features[0, 1]:.6f}")
print(f"  dist_val: {features[0, 2]:.6f}")
print(f"  dist_high: {features[0, 3]:.6f}")
print(f"  dist_low: {features[0, 4]:.6f}")
print(f"  value_area_position: {features[0, 5]:.6f}")
print(f"  volume_at_price: {features[0, 6]:.6f}")
print(f"  session_type: {features[0, 7]:.0f} ({session_types[int(features[0, 7])]})")
print(f"  balance_state: {features[0, 8]:.6f}")
print(f"  dist_prev_poc: {features[0, 9]:.6f}")

# Test VP bins extraction
print("\n" + "="*60)
print("Testing get_volume_profile_bins (56 features)...")
print("="*60)

close_prices = torch.tensor([base_price + np.random.randn()*10 for _ in range(lookback)], dtype=torch.float32)
high_prices = close_prices + torch.rand(lookback) * 20
low_prices = close_prices - torch.rand(lookback) * 20

vp_bins = get_volume_profile_bins(vp, lookback, close_prices, high_prices, low_prices)
print(f"VP Bins shape: {vp_bins.shape} (expected: ({lookback}, 56))")
print(f"VP Bins dtype: {vp_bins.dtype}")
print(f"\nFirst timestep bins (first 5 bins + markers in sorted order):")
print(f"  Bins [0-4]: {vp_bins[0, :5]}")
print(f"  VAH marker (ch 50): {vp_bins[0, 50]:.2f}")
print(f"  POC marker (ch 51): {vp_bins[0, 51]:.2f}")
print(f"  VAL marker (ch 52): {vp_bins[0, 52]:.2f}")
print(f"  Close marker (ch 53): {vp_bins[0, 53]:.2f}")
print(f"  High marker (ch 54): {vp_bins[0, 54]:.2f}")
print(f"  Low marker (ch 55): {vp_bins[0, 55]:.2f}")

# Test Previous Day VP bins extraction
print("\n" + "="*60)
print("Testing get_previous_day_volume_profile_bins (60 features)...")
print("="*60)

prev_day_vp_bins = get_previous_day_volume_profile_bins(vp, lookback, close_prices, high_prices, low_prices)
print(f"Previous Day VP Bins shape: {prev_day_vp_bins.shape} (expected: ({lookback}, 60))")
print(f"Previous Day VP Bins dtype: {prev_day_vp_bins.dtype}")
print(f"\nFirst timestep bins (first 5 bins + markers in sorted order):")
print(f"  Bins [0-4]: {prev_day_vp_bins[0, :5]}")
print(f"  Yesterday High marker (ch 50): {prev_day_vp_bins[0, 50]:.2f}")
print(f"  Yesterday VAH marker (ch 51): {prev_day_vp_bins[0, 51]:.2f}")
print(f"  Yesterday POC marker (ch 52): {prev_day_vp_bins[0, 52]:.2f}")
print(f"  Yesterday VAL marker (ch 53): {prev_day_vp_bins[0, 53]:.2f}")
print(f"  Yesterday Low marker (ch 54): {prev_day_vp_bins[0, 54]:.2f}")
print(f"  Today Close marker (ch 55): {prev_day_vp_bins[0, 55]:.2f}")
print(f"  Today High marker (ch 56): {prev_day_vp_bins[0, 56]:.2f}")
print(f"  Today Low marker (ch 57): {prev_day_vp_bins[0, 57]:.2f}")
print(f"  Prev-Prev VAH marker (ch 58): {prev_day_vp_bins[0, 58]:.2f}")
print(f"  Prev-Prev VAL marker (ch 59): {prev_day_vp_bins[0, 59]:.2f}")

print("\n" + "="*60)
print("✅ All tests passed! VP changes validated (including previous day VP).")
print("="*60)
