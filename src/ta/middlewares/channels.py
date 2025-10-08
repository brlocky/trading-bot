"""
Channels Middleware for Technical Analysis
Finds price channels using zigzag pivots and returns channel lines.
"""

from typing import List
import pandas as pd
import numpy as np
from core.trading_types import ChartInterval
from ta.technical_analysis import AnalysisDict, Line, Pivot


def _transform_price(price: float, log_scale: bool) -> float:
    """Transform price value based on scale type."""
    return np.log(price) if log_scale else price


def _inverse_transform_price(value: float, log_scale: bool) -> float:
    """Inverse transform price value based on scale type."""
    return float(np.exp(value)) if log_scale else value


def _clamp_price(val: float, price_data: pd.DataFrame) -> float:
    """Clamp price value within reasonable bounds."""
    max_price = price_data['high'].max()
    min_price = price_data['low'].min()
    height = max_price - min_price
    return min(val, max_price + height)


def _find_best_second_pivot(pivots: List[Pivot], first_pivot: Pivot, price_data: pd.DataFrame,
                            log_scale: bool, is_uptrend: bool) -> Pivot | None:
    """Find the best second pivot with minimum violations - OPTIMIZED VERSION."""
    p1 = first_pivot
    idx_p1 = pivots.index(p1)
    pivot_type = 'low' if is_uptrend else 'high'
    candidates = [p for p in pivots[idx_p1+1:] if p[2] == pivot_type]

    if not candidates:
        return None

    # Pre-compute all timestamp-to-index mappings once
    timestamp_to_idx = {ts: i for i, ts in enumerate(price_data.index)}

    # Get p1 index once
    idx1 = timestamp_to_idx[p1[0]]
    last_idx = len(price_data) - 1

    # Pre-transform all close prices once (vectorized)
    close_values = price_data['close'].values.astype(float)
    if log_scale:
        close_prices = np.log(close_values)
    else:
        close_prices = close_values

    # Transform p1 price once
    p1v = _transform_price(p1[1], log_scale)

    best_p2 = None
    min_violations = float('inf')

    for candidate in candidates:
        # Get candidate index from pre-computed mapping
        idx2 = timestamp_to_idx.get(candidate[0])
        if idx2 is None or idx2 <= idx1:
            continue

        # Transform candidate price
        p2v = _transform_price(candidate[1], log_scale)

        # Calculate slope once
        m = (p2v - p1v) / float(idx2 - idx1) if (idx2 - idx1) != 0 else 0.0

        # Vectorized violation calculation
        indices = np.arange(idx1 + 1, last_idx + 1)
        expected_values = m * (indices - idx1) + p1v
        actual_values = close_prices[idx1 + 1:last_idx + 1]

        # Count violations using vectorized operations
        if is_uptrend:
            violations = np.sum(actual_values < expected_values)
        else:
            violations = np.sum(actual_values > expected_values)

        if violations < min_violations:
            min_violations = violations
            best_p2 = candidate
            if violations == 0:
                break

    return best_p2


def _create_channel_lines(p1: Pivot, p2: Pivot, p3: Pivot, price_data: pd.DataFrame,
                          pivots: List[Pivot], log_scale: bool, is_uptrend: bool) -> List[Line]:
    """Create channel lines for both uptrend and downtrend."""
    idx1 = price_data.index.get_indexer([p1[0]])[0]
    idx2 = price_data.index.get_indexer([p2[0]])[0]
    idx3 = price_data.index.get_indexer([p3[0]])[0]
    last_time = price_data.index[-1]
    last_idx = price_data.index.get_indexer([last_time])[0]

    # Pre-compute price bounds for clamping (optimization)
    max_price = price_data['high'].max()
    min_price = price_data['low'].min()
    height = max_price - min_price
    max_clamp = max_price + height

    def clamp_price_fast(val: float) -> float:
        return min(val, max_clamp)

    # Transform prices
    p1v = _transform_price(p1[1], log_scale)
    p2v = _transform_price(p2[1], log_scale)
    p3v = _transform_price(p3[1], log_scale)

    # Calculate slope
    m = (p2v - p1v) / float(idx2 - idx1) if (idx2 - idx1) != 0 else 0.0

    if is_uptrend:
        # Calculate upper channel line
        b_upper = p3v - m * idx3

        # Lower line (trend line)
        lower_p2_val = clamp_price_fast(_inverse_transform_price(m * last_idx + (p1v - m * idx1), log_scale))
        lower_p2: Pivot = (last_time, lower_p2_val, 'low')
        lower_line: Line = (p1, lower_p2, 'channel_lower_line', None)

        # Upper line (parallel to trend line)
        upper_p2_val = clamp_price_fast(_inverse_transform_price(m * last_idx + b_upper, log_scale))
        upper_p2: Pivot = (last_time, upper_p2_val, 'high')
        upper_p1_val = clamp_price_fast(_inverse_transform_price(m * idx1 + b_upper, log_scale))
        upper_p1: Pivot = (p1[0], upper_p1_val, 'high')
        upper_line: Line = (upper_p1, upper_p2, 'channel_upper_line', None)

        return [lower_line, upper_line]
    else:
        # Downtrend
        b_lower = p3v - m * idx3

        # Upper line (trend line)
        upper_p2_val = clamp_price_fast(_inverse_transform_price(m * last_idx + (p1v - m * idx1), log_scale))
        upper_p2_dt: Pivot = (last_time, upper_p2_val, 'high')
        upper_line_dt: Line = (p1, upper_p2_dt, 'channel_upper_line', None)

        # Lower line (parallel to trend line)
        lower_p2_val = clamp_price_fast(_inverse_transform_price(m * last_idx + b_lower, log_scale))
        lower_p2_dt: Pivot = (last_time, lower_p2_val, 'low')
        lower_p1_val = clamp_price_fast(_inverse_transform_price(m * idx1 + b_lower, log_scale))
        lower_p1_dt: Pivot = (p1[0], lower_p1_val, 'low')
        lower_line_dt: Line = (lower_p1_dt, lower_p2_dt, 'channel_lower_line', None)

        return [upper_line_dt, lower_line_dt]


def channels_middleware(
        time_frame: ChartInterval,
        price_data: pd.DataFrame,
        last_Pivot: Pivot,
        analysis: AnalysisDict,
        log_scale: bool = True
) -> AnalysisDict:
    """
    Channels Middleware: Finds price channels using zigzag pivots.
    Returns lines for channel boundaries.
    """
    zigzag = analysis.get('zigzag', {})
    pivots = zigzag.get('pivots', [])
    if not pivots:
        print("No pivots found for channels middleware.")
        return {'channels': {'lines': [], 'pivots': []}}

    if len(pivots) < 2:
        print("Warning - Not enough pivots to form channels.")
        return {'channels': {'lines': [], 'pivots': []}}

    # Separate lows and highs
    lows = [p for p in pivots if p[2] == 'low']
    highs = [p for p in pivots if p[2] == 'high']
    if not lows or not highs:
        print("Warning - Not enough Lows and/or Highs.")
        return {'channels': {'lines': [], 'pivots': []}}

    # Determine trend direction using min(low) and max(high) pivots
    min_low_pivot = min(lows, key=lambda p: p[1])
    max_high_pivot = max(highs, key=lambda p: p[1])
    uptrend = min_low_pivot[0] < max_high_pivot[0]

    # Find the first pivot based on trend direction
    if uptrend:
        p1 = min(lows, key=lambda p: p[1])  # Lowest low
    else:
        p1 = max(highs, key=lambda p: p[1])  # Highest high

    # Find the best second pivot
    p2 = _find_best_second_pivot(pivots, p1, price_data, log_scale, uptrend)
    if p2 is None:
        return {'channels': {'lines': [], 'pivots': []}}

    # Find the third pivot for the opposite channel line
    idx_p1 = pivots.index(p1)
    idx_p2 = pivots.index(p2)

    if uptrend:
        # Find highest high between p1 and p2
        pivots_between = [p for p in pivots[idx_p1:idx_p2+1] if p[2] == 'high']
        if pivots_between:
            p3 = max(pivots_between, key=lambda p: p[1])
        else:
            p3 = max(highs, key=lambda p: p[1])
    else:
        # Find lowest low between p1 and p2
        pivots_between = [p for p in pivots[idx_p1:idx_p2+1] if p[2] == 'low']
        if pivots_between:
            p3 = min(pivots_between, key=lambda p: p[1])
        else:
            p3 = min(lows, key=lambda p: p[1])

    # Create channel lines
    channel_lines = _create_channel_lines(p1, p2, p3, price_data, pivots, log_scale, uptrend)
    pivots_used = [p1, p2, p3]

    return {'channels': {'lines': channel_lines, 'pivots': pivots_used}}
