
"""
Global Volume Profile Middleware for Technical Analysis
Returns AnalysisDict compatible with other middlewares.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from src.ta.technical_analysis import AnalysisDict, ChartInterval, VolumeProfileLine


class VolumeProfileKeyLevels(TypedDict):
    poc_price: float
    poc_volume: float
    vah_price: float
    val_price: float
    vah_volume: float
    val_volume: float


def calculate_volume_profile_histogram(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int = 255,
    bin_size: Optional[float] = None
) -> Tuple[np.ndarray, List[float], float, float, VolumeProfileKeyLevels]:
    """
    Shared function to calculate volume profile histogram with POC, VAH, VAL.

    Args:
        prices: Array of prices
        volumes: Array of volumes
        bins: Number of bins if bin_size is None
        bin_size: Optional size of each price bin

    Returns:
        Tuple of (bin_edges, bin_volumes, min_price, max_price, key_levels)
        key_levels contains: {'poc_price': float, 'poc_volume': float,
                             'vah_price': float, 'val_price': float,
                             'vah_volume': float, 'val_volume': float}
    """
    # Input validation
    if prices.size == 0 or volumes.size == 0:
        raise ValueError("Price and volume arrays must not be empty.")
    if not np.issubdtype(prices.dtype, np.number) or not np.issubdtype(volumes.dtype, np.number):
        raise ValueError("Price and volume arrays must be numeric.")

    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    price_range = max_price - min_price

    # Skip if no price movement
    if price_range == 0.0:
        total_vol = float(np.sum(volumes))
        key_levels: VolumeProfileKeyLevels = {
            'poc_price': min_price,
            'poc_volume': total_vol,
            'vah_price': min_price,
            'val_price': min_price,
            'vah_volume': total_vol,
            'val_volume': total_vol
        }
        return np.array([min_price, max_price]), [total_vol], min_price, max_price, key_levels

    if bin_size is None:
        bin_size = price_range / bins

    bin_edges = np.arange(min_price, max_price + bin_size, bin_size)
    bin_indices = np.digitize(prices, bin_edges) - 1
    bin_volumes = [float(np.sum(volumes[bin_indices == i])) for i in range(len(bin_edges) - 1)]

    # Calculate POC (Point of Control), VAH (Value Area High), and VAL (Value Area Low)
    poc_idx = np.argmax(bin_volumes)
    poc_price_start = float(bin_edges[poc_idx])
    poc_price_end = float(bin_edges[poc_idx + 1])
    poc_price_mid = (poc_price_start + poc_price_end) / 2
    poc_volume = bin_volumes[poc_idx]

    # Calculate Value Area (70% of total volume)
    total_volume = sum(bin_volumes)
    value_area_volume = total_volume * 0.70

    # Find VAH and VAL by expanding from POC until we reach 70% of volume
    accumulated_volume = bin_volumes[poc_idx]
    upper_idx = poc_idx
    lower_idx = poc_idx

    # Standard volume profile expansion logic
    while accumulated_volume < value_area_volume and (upper_idx < len(bin_volumes) - 1 or lower_idx > 0):
        # Check which direction to expand (higher volume gets priority)
        upper_vol = bin_volumes[upper_idx + 1] if upper_idx < len(bin_volumes) - 1 else 0
        lower_vol = bin_volumes[lower_idx - 1] if lower_idx > 0 else 0

        if upper_vol >= lower_vol and upper_idx < len(bin_volumes) - 1:
            upper_idx += 1
            accumulated_volume += bin_volumes[upper_idx]
        elif lower_idx > 0:
            lower_idx -= 1
            accumulated_volume += bin_volumes[lower_idx]
        else:
            break

    # Calculate VAH and VAL prices
    vah_price = float(bin_edges[upper_idx + 1]) if upper_idx < len(bin_edges) - 1 else float(bin_edges[upper_idx])
    val_price = float(bin_edges[lower_idx])
    vah_volume = bin_volumes[upper_idx] if upper_idx < len(bin_volumes) else 0.0
    val_volume = bin_volumes[lower_idx] if lower_idx < len(bin_volumes) else 0.0

    key_levels: VolumeProfileKeyLevels = {
        'poc_price': poc_price_mid,
        'poc_volume': poc_volume,
        'vah_price': vah_price,
        'val_price': val_price,
        'vah_volume': vah_volume,
        'val_volume': val_volume
    }

    return bin_edges, bin_volumes, min_price, max_price, key_levels


def global_volume_profile_middleware(
    time_frame: ChartInterval,
    price_data: pd.DataFrame,
    analysis: AnalysisDict,
    useLogScale: bool = True,
    bin_size: Optional[float] = None,
    bins: int = 255
) -> AnalysisDict:
    """
    Calculates volume profile as a histogram of volume at price bins.
    Returns AnalysisDict with 'volume_profile' key.
    Args:
        price_data: DataFrame with 'close' and 'volume' columns.
        bin_size: Optional, size of each price bin. If None, auto-calculated.
        bins: Number of bins if bin_size is None.
    Returns:
        AnalysisDict: {'volume_profile': {'lines': [...], 'pivots': []}}
    """
    prices = price_data['close'].to_numpy(dtype=float)
    volumes = price_data['volume'].to_numpy(dtype=float)

    # Use shared calculation function
    bin_edges, bin_volumes, min_price, max_price, key_levels = calculate_volume_profile_histogram(
        prices, volumes, bins, bin_size
    )

    vp: List[VolumeProfileLine] = []
    max_vol = max(bin_volumes) if bin_volumes else 1.0

    for i in range(len(bin_edges) - 1):
        bin_start = float(bin_edges[i])
        bin_end = float(bin_edges[i + 1])
        bin_vol = bin_volumes[i]
        norm_vol = bin_vol / max_vol if max_vol > 0 else 0.0
        # vp: ((bin_start, bin_end), bin_vol, norm_vol)
        vp.append(((bin_start, bin_end), bin_vol, norm_vol))

    # Create key levels as volume profile lines
    poc_line = ((key_levels['poc_price'], key_levels['poc_price']), key_levels['poc_volume'], 1.0)
    vah_norm_vol = key_levels['vah_volume'] / max_vol if max_vol > 0 else 0.0
    vah_line = ((key_levels['vah_price'], key_levels['vah_price']), key_levels['vah_volume'], vah_norm_vol)
    val_norm_vol = key_levels['val_volume'] / max_vol if max_vol > 0 else 0.0
    val_line = ((key_levels['val_price'], key_levels['val_price']), key_levels['val_volume'], val_norm_vol)

    # Add key levels to vp
    vp.extend([poc_line, vah_line, val_line])

    return {
        'volume_profile': {'vp': vp},
    }
