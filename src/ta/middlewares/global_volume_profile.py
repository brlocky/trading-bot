
"""
Global Volume Profile Middleware for Technical Analysis
Returns AnalysisDict compatible with other middlewares.
"""
import numpy as np
import pandas as pd
from typing import Optional, List
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from core.trading_types import ChartInterval
from ta.technical_analysis import AnalysisDict, Pivot, VolumeProfileLine


class VolumeProfileKeyLevels(TypedDict):
    poc_price: float
    poc_volume: float
    vah_price: float
    val_price: float
    vah_volume: float
    val_volume: float


def calculate_key_volume_levels_fast(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int = 100
) -> VolumeProfileKeyLevels:
    """
    OPTIMIZED: Calculate only POC, VAH, VAL with GPU acceleration.
    Automatically uses GPU if available, falls back to CPU.
    """
    if prices.size == 0 or volumes.size == 0:
        raise ValueError("Price and volume arrays must not be empty.")

    # GPU-accelerated operations
    try:
        from ta.gpu_utils import GPUArrayOps
        ops = GPUArrayOps()  # Auto-detects GPU
    except ImportError:
        # Fallback to plain numpy if gpu_utils not available
        class FallbackOps:
            def asarray(self, x): return np.asarray(x)
            def asnumpy(self, x): return np.asarray(x)
            def min(self, x, **kw): return np.min(x, **kw)
            def max(self, x, **kw): return np.max(x, **kw)
            def sum(self, x, **kw): return np.sum(x, **kw)
            def arange(self, *args, **kw): return np.arange(*args, **kw)
            def digitize(self, x, bins): return np.digitize(x, bins)
            def clip(self, x, a, b): return np.clip(x, a, b)
            def bincount(self, x, **kw): return np.bincount(x, **kw)
            def argmax(self, x, **kw): return np.argmax(x, **kw)
        ops = FallbackOps()

    # Move to GPU
    prices_gpu = ops.asarray(prices)
    volumes_gpu = ops.asarray(volumes)

    min_price = float(ops.min(prices_gpu))
    max_price = float(ops.max(prices_gpu))
    price_range = max_price - min_price

    # Handle single price case
    if price_range == 0.0:
        total_vol = float(ops.sum(volumes_gpu))
        return {
            'poc_price': min_price,
            'poc_volume': total_vol,
            'vah_price': min_price,
            'val_price': min_price,
            'vah_volume': total_vol,
            'val_volume': total_vol
        }

    # GPU-accelerated binning
    bin_size = price_range / bins
    bin_edges = ops.arange(min_price, max_price + bin_size, bin_size)

    # Vectorized binning on GPU
    bin_indices = ops.digitize(prices_gpu, bin_edges) - 1
    bin_indices = ops.clip(bin_indices, 0, len(bin_edges) - 2)

    # Fast histogram calculation on GPU
    bin_volumes = ops.bincount(bin_indices, weights=volumes_gpu, minlength=len(bin_edges)-1)

    # Move results back to CPU for processing
    bin_volumes_cpu = ops.asnumpy(bin_volumes)
    bin_edges_cpu = ops.asnumpy(bin_edges)

    # Find POC
    poc_idx = int(ops.argmax(bin_volumes))
    poc_price = (bin_edges_cpu[poc_idx] + bin_edges_cpu[poc_idx + 1]) / 2
    poc_volume = float(bin_volumes_cpu[poc_idx])

    # Fast Value Area calculation (70% of total volume)
    total_volume = float(ops.sum(bin_volumes))
    value_area_volume = total_volume * 0.70

    # Vectorized expansion from POC (CPU is fine for this small loop)
    accumulated_volume = bin_volumes_cpu[poc_idx]
    upper_idx = poc_idx
    lower_idx = poc_idx

    while accumulated_volume < value_area_volume and (upper_idx < len(bin_volumes_cpu) - 1 or lower_idx > 0):
        upper_vol = bin_volumes_cpu[upper_idx + 1] if upper_idx < len(bin_volumes_cpu) - 1 else 0
        lower_vol = bin_volumes_cpu[lower_idx - 1] if lower_idx > 0 else 0

        if upper_vol >= lower_vol and upper_idx < len(bin_volumes_cpu) - 1:
            upper_idx += 1
            accumulated_volume += bin_volumes_cpu[upper_idx]
        elif lower_idx > 0:
            lower_idx -= 1
            accumulated_volume += bin_volumes_cpu[lower_idx]
        else:
            break

    # Calculate final VAH/VAL
    vah_price = bin_edges_cpu[upper_idx + 1] if upper_idx < len(bin_edges_cpu) - 1 else bin_edges_cpu[upper_idx]
    val_price = bin_edges_cpu[lower_idx]
    vah_volume = float(bin_volumes_cpu[upper_idx]) if upper_idx < len(bin_volumes_cpu) else 0.0
    val_volume = float(bin_volumes_cpu[lower_idx]) if lower_idx < len(bin_volumes_cpu) else 0.0

    return {
        'poc_price': float(poc_price),
        'poc_volume': poc_volume,
        'vah_price': float(vah_price),
        'val_price': float(val_price),
        'vah_volume': vah_volume,
        'val_volume': val_volume
    }


def global_volume_profile_middleware(
    time_frame: ChartInterval,
    price_data: pd.DataFrame,
    last_Pivot: Pivot,
    analysis: AnalysisDict,
    useLogScale: bool = True,
    bin_size: Optional[float] = None,
    bins: int = 100
) -> AnalysisDict:
    """
    OPTIMIZED: Calculates only key volume levels (POC, VAH, VAL).
    Much faster since we skip unnecessary histogram generation.
    """
    prices = price_data['close'].to_numpy(dtype=float)
    volumes = price_data['volume'].to_numpy(dtype=float)

    # Use optimized key levels calculation
    key_levels = calculate_key_volume_levels_fast(prices, volumes, bins)

    # Create simplified volume profile with just key levels
    vp: List[VolumeProfileLine] = []

    # Only add the key levels we actually care about
    poc_line = ((key_levels['poc_price'], key_levels['poc_price']), key_levels['poc_volume'], 1.0)
    vah_line = ((key_levels['vah_price'], key_levels['vah_price']), key_levels['vah_volume'], 0.8)
    val_line = ((key_levels['val_price'], key_levels['val_price']), key_levels['val_volume'], 0.8)

    vp.extend([poc_line, vah_line, val_line])

    return {
        'volume_profile': {'vp': vp},
    }


# Backwards compatibility wrapper
def calculate_volume_profile_histogram(prices, volumes, bins=100, bin_size=None):
    """Backwards compatibility wrapper for the old function name."""
    key_levels = calculate_key_volume_levels_fast(prices, volumes, bins)
    return None, [], 0.0, 0.0, key_levels
