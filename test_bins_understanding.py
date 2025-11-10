"""Test to understand what vp_obj.bins actually contains."""
import torch
import sys
sys.path.insert(0, 'src')

from features.enhanced_volume_profile import EnhancedVolumeProfile
from datetime import datetime, timedelta

# Create VP object
vp = EnhancedVolumeProfile(n_bins=10, lookback_window=5, device="cpu")

# Simulate some price data
base_date = datetime(2024, 1, 1, 0, 0)
for i in range(5):
    timestamp = base_date + timedelta(hours=i)
    open_price = 100.0 + i
    high_price = 102.0 + i
    low_price = 99.0 + i
    close_price = 101.0 + i
    volume = 1000.0
    
    vp.update(timestamp, open_price, high_price, low_price, close_price, volume)

# Now check what bins contains
print("=" * 80)
print("UNDERSTANDING VP BINS")
print("=" * 80)
print(f"\nvp.bins shape: {vp.bins.shape}")
print(f"vp.bins type: {type(vp.bins)}")
print(f"vp.bins contents:\n{vp.bins}")
print(f"\nFirst bin (price_min): {vp.bins[0].item()}")
print(f"Last bin (price_max): {vp.bins[-1].item()}")

print(f"\nvp.weights shape: {vp.weights.shape}")
print(f"vp.weights type: {type(vp.weights)}")
print(f"vp.weights contents:\n{vp.weights}")
print(f"vp.weights sum: {vp.weights.sum().item()}")

# Get bins history
bins_history = vp.get_bins_history(5)
print(f"\nbins_history shape: {bins_history.shape}")
print(f"bins_history contents:\n{bins_history}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("vp.bins = PRICE bin edges (linspace from price_min to price_max)")
print("vp.weights = VOLUME distribution (normalized, sums to 1.0)")
print("get_bins_history() returns = VOLUME weights history (not prices!)")
print("=" * 80)
