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


def _find_best_second_pivot(pivots: List[Pivot], first_pivot: Pivot, price_data: pd.DataFrame,
                            log_scale: bool, is_uptrend: bool, min_candles_after: int = 5) -> Pivot | None:
    """Find the best second pivot with minimum violations - OPTIMIZED VERSION.

    Args:
        min_candles_after: Minimum number of candles required after p2 to validate the channel (default: 5)
    """
    p1 = first_pivot
    idx_p1 = pivots.index(p1)
    # For uptrend, look for LL or HL pivots; for downtrend, look for HH or LH pivots
    if is_uptrend:
        candidates = [p for p in pivots[idx_p1+1:] if p[3] in ('LL', 'HL')]
    else:
        candidates = [p for p in pivots[idx_p1+1:] if p[3] in ('HH', 'LH')]

    if not candidates:
        return None

    # Pre-compute all timestamp-to-index mappings once
    timestamp_to_idx = {ts: i for i, ts in enumerate(price_data.index)}
    last_idx = len(price_data) - 1

    # Filter out candidates that are too close to the end (need candles after to validate)
    candidates = [c for c in candidates
                  if timestamp_to_idx.get(c[0], last_idx) <= last_idx - min_candles_after]

    if not candidates:
        return None

    # Get p1 index once
    idx1 = timestamp_to_idx[p1[0]]

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


def _find_best_parallel_pivot(pivots: List[Pivot], p1: Pivot, p2: Pivot,
                              price_data: pd.DataFrame, log_scale: bool,
                              is_uptrend: bool) -> Pivot | None:
    """
    Find the best pivot for the parallel channel line with minimum wick violations.

    For uptrend: Find best HH between p1 and p2 where parallel line doesn't touch wicks
    For downtrend: Find best LL between p1 and p2 where parallel line doesn't touch wicks

    Strategy: Test each candidate and count wick violations on the parallel line.
    Choose the one with minimum violations (ideally zero).
    """
    idx_p1 = pivots.index(p1)
    idx_p2 = pivots.index(p2)

    # Get candidates between p1 and p2
    if is_uptrend:
        # For uptrend, we want HH or LH pivots (highs) for the upper parallel line
        candidates = [p for p in pivots[idx_p1:idx_p2+1] if p[3] in ('HH', 'LH')]
    else:
        # For downtrend, we want LL or HL pivots (lows) for the lower parallel line
        candidates = [p for p in pivots[idx_p1:idx_p2+1] if p[3] in ('LL', 'HL')]

    if not candidates:
        return None

    # Pre-compute timestamp-to-index mappings
    timestamp_to_idx = {ts: i for i, ts in enumerate(price_data.index)}
    idx1 = timestamp_to_idx[p1[0]]
    idx2 = timestamp_to_idx[p2[0]]

    # Transform p1 and p2 for slope calculation
    p1v = _transform_price(p1[1], log_scale)
    p2v = _transform_price(p2[1], log_scale)

    # Calculate trendline slope
    m = (p2v - p1v) / float(idx2 - idx1) if (idx2 - idx1) != 0 else 0.0

    # Pre-transform wick prices (high for uptrend, low for downtrend)
    if is_uptrend:
        wick_values = price_data['high'].values.astype(float)
    else:
        wick_values = price_data['low'].values.astype(float)

    if log_scale:
        wick_prices = np.log(wick_values)
    else:
        wick_prices = wick_values

    best_p3 = None
    min_violations = float('inf')

    # Test each candidate
    for candidate in candidates:
        idx3 = timestamp_to_idx.get(candidate[0])
        if idx3 is None or idx3 < idx1 or idx3 > idx2:
            continue

        # Transform candidate price
        p3v = _transform_price(candidate[1], log_scale)

        # Calculate parallel line intercept (offset from trendline)
        b_parallel = p3v - m * idx3

        # Check violations on parallel line from p1 to p2 (only between trendline pivots)
        indices = np.arange(idx1, idx2 + 1)
        expected_values = m * indices + b_parallel
        actual_values = wick_prices[idx1:idx2 + 1]

        # Count wick violations
        if is_uptrend:
            # For uptrend upper line, wicks should not go above the line
            violations = np.sum(actual_values > expected_values)
        else:
            # For downtrend lower line, wicks should not go below the line
            violations = np.sum(actual_values < expected_values)

        # Keep track of best candidate
        if violations < min_violations:
            min_violations = violations
            best_p3 = candidate
            if violations == 0:
                break  # Perfect fit found!

    return best_p3


def _create_channel_lines(p1: Pivot, p2: Pivot, p3: Pivot, price_data: pd.DataFrame,
                          pivots: List[Pivot], log_scale: bool, is_uptrend: bool) -> List[Line]:
    """Create channel lines for both uptrend and downtrend.

    Simplified: Calculate slope using _transform_price, but work directly with actual prices
    in normal scale. No inverse transform needed.
    """
    idx1 = price_data.index.get_indexer([p1[0]])[0]
    idx2 = price_data.index.get_indexer([p2[0]])[0]
    idx3 = price_data.index.get_indexer([p3[0]])[0]
    last_idx = len(price_data) - 1

    # Get actual timestamp from DataFrame for the last candle
    import pandas as pd
    time_at_last: pd.Timestamp = price_data.index[last_idx]  # type: ignore

    # Pre-compute price bounds for clamping (optimization)
    max_price = price_data['high'].max()
    min_price = price_data['low'].min()
    height = max_price - min_price
    max_clamp = max_price + height

    def clamp_price_fast(val: float) -> float:
        return min(val, max_clamp)

    # Transform prices for slope calculation
    p1v = _transform_price(p1[1], log_scale)
    p2v = _transform_price(p2[1], log_scale)
    p3v = _transform_price(p3[1], log_scale)

    # Calculate slope in transformed space (log if log_scale=True, normal otherwise)
    m = (p2v - p1v) / float(idx2 - idx1) if (idx2 - idx1) != 0 else 0.0

    if is_uptrend:
        # Lower line (trend line) - extend from p1 to end in transformed space
        # The trendline equation is: price = p1v + m * (idx - idx1)
        # At idx1: price = p1v + m * 0 = p1v ✓
        # At last_idx: price = p1v + m * (last_idx - idx1)
        lower_p2_transformed = p1v + m * (last_idx - idx1)
        lower_p2_val = clamp_price_fast(_inverse_transform_price(lower_p2_transformed, log_scale))
        lower_p2: Pivot = (time_at_last, lower_p2_val, 0, None)  # Use actual timestamp
        lower_line: Line = (p1, lower_p2, 'channel_lower_line', None)

        # Upper line (parallel to trend line, passing through p3)
        # Start from p3 and extend to the end
        # Equation: price = p3v + m * (idx - idx3)

        # Calculate price at last position
        upper_p2_transformed = p3v + m * (last_idx - idx3)
        upper_p2_val = clamp_price_fast(_inverse_transform_price(upper_p2_transformed, log_scale))

        # Start from p3 (exact time and price)
        upper_p2: Pivot = (time_at_last, upper_p2_val, 0, None)
        upper_line: Line = (p3, upper_p2, 'channel_upper_line', None)

        return [lower_line, upper_line]
    else:
        # Downtrend - Upper line is the trend line
        # Equation: price = p1v + m * (idx - idx1)
        upper_p2_transformed = p1v + m * (last_idx - idx1)
        upper_p2_val = clamp_price_fast(_inverse_transform_price(upper_p2_transformed, log_scale))
        upper_p2_dt: Pivot = (time_at_last, upper_p2_val, 0, None)  # Use actual timestamp
        upper_line_dt: Line = (p1, upper_p2_dt, 'channel_upper_line', None)

        # Lower line (parallel to trend line, passing through p3)
        # Start from p3 and extend to the end
        lower_p2_transformed = p3v + m * (last_idx - idx3)
        lower_p2_val = clamp_price_fast(_inverse_transform_price(lower_p2_transformed, log_scale))

        # Start from p3 (exact time and price)
        lower_p2: Pivot = (time_at_last, lower_p2_val, 0, None)
        lower_line: Line = (p3, lower_p2, 'channel_lower_line', None)

        return [upper_line_dt, lower_line]


def _find_single_channel(pivots: List[Pivot], price_data: pd.DataFrame,
                         log_scale: bool) -> tuple[List[Line], List[Pivot]] | None:
    """
    Extract a single channel from pivots. Returns (lines, pivots_used) or None.
    This is the core channel-finding logic extracted for reuse.
    """
    if len(pivots) < 2:
        return None

    # Separate lows and highs using swing types
    lows = [p for p in pivots if p[3] in ('LL', 'HL')]  # Low pivots
    highs = [p for p in pivots if p[3] in ('HH', 'LH')]  # High pivots
    if not lows or not highs:
        return None

    # Determine trend direction using min(low) and max(high) pivots
    min_low_pivot = min(lows, key=lambda p: p[1])
    max_high_pivot = max(highs, key=lambda p: p[1])
    uptrend = min_low_pivot[0] < max_high_pivot[0]

    # Find the first pivot based on trend direction
    if uptrend:
        p1 = min(lows, key=lambda p: p[1])  # Lowest low
    else:
        p1 = max(highs, key=lambda p: p[1])  # Highest high

    # Find the best second pivot (p2) with minimum body close violations
    p2 = _find_best_second_pivot(pivots, p1, price_data, log_scale, uptrend)
    if p2 is None:
        return None

    # NEW: Find the best third pivot (p3) for parallel line with minimum WICK violations
    # This is the key improvement - test each candidate to find the one that creates
    # the best parallel channel line (no wicks touching the line)
    p3 = _find_best_parallel_pivot(pivots, p1, p2, price_data, log_scale, uptrend)

    # Fallback: If no good p3 found, use the extreme pivot as before
    if p3 is None:
        idx_p1 = pivots.index(p1)
        idx_p2 = pivots.index(p2)
        if uptrend:
            pivots_between = [p for p in pivots[idx_p1:idx_p2+1] if p[3] in ('HH', 'LH')]
            if pivots_between:
                p3 = max(pivots_between, key=lambda p: p[1])
            else:
                p3 = max(highs, key=lambda p: p[1])
        else:
            pivots_between = [p for p in pivots[idx_p1:idx_p2+1] if p[3] in ('LL', 'HL')]
            if pivots_between:
                p3 = min(pivots_between, key=lambda p: p[1])
            else:
                p3 = min(lows, key=lambda p: p[1])

    # Create channel lines
    channel_lines = _create_channel_lines(p1, p2, p3, price_data, pivots, log_scale, uptrend)
    pivots_used = [p1, p2, p3]

    return (channel_lines, pivots_used)


def channels_middleware(
        time_frame: ChartInterval,
        price_data: pd.DataFrame,
        last_Pivot: Pivot,
        analysis: AnalysisDict,
        log_scale: bool = True,
        max_sub_channels: int = 3
) -> AnalysisDict:
    """
    Channels Middleware: Finds price channels using zigzag pivots.
    Now supports recursive sub-channels starting from p2 of previous channel.

    Args:
        max_sub_channels: Maximum number of sub-channels to create (default: 3)
                         Set to 1 for original behavior (no sub-channels)
    """
    zigzag = analysis.get('zigzag', {})
    pivots = zigzag.get('pivots', [])
    if not pivots:
        print("No pivots found for channels middleware.")
        return {'channels': {'lines': [], 'pivots': []}}

    all_channel_lines = []
    all_pivots_used = []

    # Find channels recursively
    remaining_pivots = pivots
    for i in range(max_sub_channels):
        # Try to find a channel from remaining pivots
        result = _find_single_channel(remaining_pivots, price_data, log_scale)

        if result is None:
            break  # No more channels can be found

        channel_lines, pivots_used = result
        all_channel_lines.extend(channel_lines)
        all_pivots_used.extend(pivots_used)

        # For next iteration, start from p2 onwards (remove pivots before p2)
        p2 = pivots_used[1]  # p2 is the second pivot in pivots_used
        try:
            idx_p2 = remaining_pivots.index(p2)
            # Keep pivots from p2 onwards for next sub-channel
            remaining_pivots = remaining_pivots[idx_p2:]

            # Need at least 3 pivots to form another channel
            if len(remaining_pivots) < 3:
                break
        except (ValueError, IndexError):
            break  # Can't find p2 or not enough pivots

    if not all_channel_lines:
        return {'channels': {'lines': [], 'pivots': []}}

    return {'channels': {'lines': all_channel_lines, 'pivots': all_pivots_used}}
