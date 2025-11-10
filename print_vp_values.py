"""
Print actual Volume Profile values from the trading environment
"""
import pandas as pd
import numpy as np
from src.utils.indicator_utils import add_indicators
from src.features.enhanced_volume_profile import EnhancedVolumeProfile

# Load data
symbol = 'BTCUSDT'
timeframe = '5m'
data_path = f'data/binance-{symbol}-{timeframe}.pkl'
df = pd.read_pickle(data_path)

print(f'Loaded {len(df)} rows for {symbol} {timeframe}\n')

# Prepare data
df['date'] = pd.to_datetime(df['date_close'])
add_indicators(df)
df = df.dropna().reset_index(drop=True)

# Use a subset - start from a later point so we have previous day data
# Let's start from index 51000 (about 3.5 days later at 5min bars = ~1000 bars per day)
test_df = df.iloc[51000:51100].copy().reset_index(drop=True)

# Initialize Volume Profile
vp = EnhancedVolumeProfile(n_bins=50, lookback_window=288, device='cpu')

print("=" * 100)
print("VOLUME PROFILE VALUES - 100 bars from index 51000 (with previous day data)")
print("=" * 100)

# Process bars and print VP features
for i in range(len(test_df)):
    row = test_df.iloc[i]

    # Update VP
    features = vp.update(
        timestamp=row['date'],
        open_price=row['open'],
        high_price=row['high'],
        low_price=row['low'],
        close_price=row['close'],
        volume=row['volume']
    )

    # Print every 10 bars
    if i % 10 == 0 or i < 5:
        print(f"\n{'='*100}")
        print(f"Bar {i}: {row['date']} | Close: ${row['close']:.2f}")
        print(f"{'='*100}")

        print(f"\n📊 Volume Profile Features (10 values):")
        print(f"   [0] Distance to POC:           {features[0].item():>8.6f}")
        print(f"   [1] Distance to Prev Day VAH:  {features[1].item():>8.6f}")
        print(f"   [2] Distance to Prev Day VAL:  {features[2].item():>8.6f}")
        print(f"   [3] Distance to Prev Day POC:  {features[3].item():>8.6f}")
        print(f"   [4] Inside Value Area:         {features[4].item():>8.1f} {'✓ YES' if features[4].item() == 1.0 else '✗ NO'}")
        print(f"   [5] Above Value Area:          {features[5].item():>8.1f} {'✓ YES' if features[5].item() == 1.0 else '✗ NO'}")
        print(f"   [6] Distance to Prev Day High: {features[6].item():>8.6f}")
        print(f"   [7] Distance to Prev Day Low:  {features[7].item():>8.6f}")
        print(f"   [8] Distance to Weekly POC:    {features[8].item():>8.6f}")
        print(f"   [9] Local Volume Concentration:{features[9].item():>8.6f}")

        # Show actual VP reference levels
        print(f"\n📈 Reference Levels:")
        if vp.prev_day_vah is not None:
            print(f"   Previous Day VAH: ${vp.prev_day_vah:.2f}")
            print(f"   Previous Day VAL: ${vp.prev_day_val:.2f}")
            print(f"   Previous Day POC: ${vp.prev_day_poc:.2f}")
        else:
            print(f"   Previous Day levels: Not yet calculated (first session)")

        if vp.prev_day_high is not None:
            print(f"   Previous Day High: ${vp.prev_day_high:.2f}")
            print(f"   Previous Day Low:  ${vp.prev_day_low:.2f}")

        if vp.prev_week_poc is not None:
            print(f"   Previous Week POC: ${vp.prev_week_poc:.2f}")

        # Show current POC
        if vp.count >= 10 and vp.weights.sum() > 0:
            poc_bin = vp.weights.argmax()
            poc_price = (vp.bins[poc_bin] + vp.bins[poc_bin + 1]) / 2
            print(f"   Current POC: ${poc_price:.2f}")

        # Volume distribution info
        print(f"\n📉 Volume Distribution:")
        print(f"   Total bars in rolling window: {vp.count}")
        print(f"   Number of bins: {vp.n_bins}")
        print(f"   Current session date: {vp.current_session_date}")
        if vp.weights.sum() > 0:
            max_weight = vp.weights.max().item()
            print(f"   Max bin weight: {max_weight:.4f}")
            print(f"   Non-zero bins: {(vp.weights > 0).sum().item()}")

# Print final summary
print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print(f"Total bars processed: {len(test_df)}")
print(f"VP lookback window: {vp.lookback_window}")
print(f"VP bins: {vp.n_bins}")
print(f"Feature history size: {vp.feature_count}")
print(f"\nNote: Features normalize distances by price to make them scale-invariant")
print(f"      Negative distance = below level, Positive = above level")
