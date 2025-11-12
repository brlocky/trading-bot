"""
Visible Range Volume Profile - Simple VP for Lookback Window

Calculates volume profile using only the visible price range (lookback window).
Agent can directly map VP bins to visible prices.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class VisibleRangeVP:
    """
    Volume Profile calculated from visible price range only.

    Much simpler than session-based VP:
    - No session detection
    - No circular buffers
    - Just calculate VP from lookback window data
    - Direct price-to-bin mapping for agent
    """

    def __init__(self, n_bins: int = 50):
        """
        Initialize Visible Range Volume Profile.

        Args:
            n_bins: Number of price bins for volume distribution
        """
        self.n_bins = n_bins

    def calculate_vp(self, price_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate volume profile for visible range.

        Args:
            price_data: DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
                       Should be the lookback window data (e.g., last 288 bars)

        Returns:
            vp_bins: [n_bins] array with volume distribution, normalized [0, 1]
            levels: dict with {'vah': float, 'val': float, 'poc': float, 'high': float, 'low': float}
        """
        if len(price_data) == 0:
            return np.zeros(self.n_bins, dtype=np.float32), {
                'vah': 0.0, 'val': 0.0, 'poc': 0.0, 'high': 0.0, 'low': 0.0
            }

        # Get price range from visible window
        visible_low = float(price_data['low'].min())
        visible_high = float(price_data['high'].max())

        if visible_high <= visible_low:
            # No price movement - return zeros
            return np.zeros(self.n_bins, dtype=np.float32), {
                'vah': visible_high, 'val': visible_low, 'poc': visible_low,
                'high': visible_high, 'low': visible_low
            }

        bin_size = (visible_high - visible_low) / self.n_bins

        # Calculate volume distribution across bins
        vp_bins = np.zeros(self.n_bins, dtype=np.float32)

        for _, row in price_data.iterrows():
            price = float(row['close'])
            volume = float(row['volume'])

            # Find which bin this price falls into
            bin_idx = int((price - visible_low) / bin_size)
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))  # Clamp to valid range

            # Accumulate volume
            vp_bins[bin_idx] += volume

        # Normalize to [0, 1]
        max_vol = vp_bins.max()
        if max_vol > 0:
            vp_bins_normalized = vp_bins / max_vol
        else:
            vp_bins_normalized = vp_bins

        # Calculate key levels (VAH/VAL/POC) from unnormalized bins
        levels = self._calculate_levels(vp_bins, visible_low, bin_size)
        levels['high'] = visible_high
        levels['low'] = visible_low

        return vp_bins_normalized, levels

    def _calculate_levels(self, vp_bins: np.ndarray, price_low: float,
                          bin_size: float) -> Dict[str, float]:
        """
        Calculate VAH/VAL/POC from volume profile.

        Args:
            vp_bins: Unnormalized volume bins
            price_low: Lowest price in visible range
            bin_size: Size of each price bin

        Returns:
            dict with 'vah', 'val', 'poc' keys
        """
        total_volume = vp_bins.sum()

        if total_volume == 0:
            # No volume - return mid-range
            mid_price = price_low + (self.n_bins * bin_size / 2)
            return {'vah': mid_price, 'val': mid_price, 'poc': mid_price}

        # Find POC (Point of Control - highest volume bin)
        poc_idx = int(np.argmax(vp_bins))
        poc_price = price_low + (poc_idx + 0.5) * bin_size  # Center of bin

        # Find Value Area (70% of volume around POC)
        target_volume = total_volume * 0.70
        accumulated_volume = float(vp_bins[poc_idx])

        va_low_idx = poc_idx
        va_high_idx = poc_idx

        # Expand around POC until we reach 70% volume
        while accumulated_volume < target_volume:
            # Check which direction has more volume
            vol_below = float(vp_bins[va_low_idx - 1]) if va_low_idx > 0 else 0.0
            vol_above = float(vp_bins[va_high_idx + 1]) if va_high_idx < self.n_bins - 1 else 0.0

            if vol_below > vol_above and va_low_idx > 0:
                # Expand downward
                va_low_idx -= 1
                accumulated_volume += float(vp_bins[va_low_idx])
            elif va_high_idx < self.n_bins - 1:
                # Expand upward
                va_high_idx += 1
                accumulated_volume += float(vp_bins[va_high_idx])
            else:
                # Can't expand further
                break

        # Convert bin indices to prices
        val_price = price_low + va_low_idx * bin_size  # Bottom of VA
        vah_price = price_low + (va_high_idx + 1) * bin_size  # Top of VA

        return {
            'vah': float(vah_price),
            'val': float(val_price),
            'poc': float(poc_price)
        }
