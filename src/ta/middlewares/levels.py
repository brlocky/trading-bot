"""
Levels Detection Middleware for Technical Analysis
Detects significant price levels such as support and resistance
"""
import pandas as pd
from typing import List, cast
from core.trading_types import ChartInterval
from ta.technical_analysis import Line, AnalysisDict, LineType, Pivot


def levels_middleware(
    time_frame: ChartInterval,
    price_data: pd.DataFrame,
    last_Pivot: Pivot,
    analysis: AnalysisDict,
    useLogScale: bool = True,
) -> AnalysisDict:
    """
    Detects significant price levels:
    1. Untouched closes - Close prices that were never revisited
    2. Period high/low - Absolute highest and lowest prices in the dataset

    Args:
        time_frame: Chart interval (e.g., '1h', 'D')
        price_data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        analysis: Existing analysis results from other middlewares
        useLogScale: Whether to use logarithmic scale for calculations
        touch_tolerance: Percentage tolerance for considering a level "touched"

    Returns:
        AnalysisDict: Dictionary containing detected levels as horizontal lines
    """

    if len(price_data) < 2:
        return {'levels': {'lines': []}}

    levels_lines: List[Line] = []
    times = price_data.index
    closes = price_data['close'].values
    highs = price_data['high'].values
    lows = price_data['low'].values
    opens = price_data['open'].values
    volumes = price_data['volume'].values

    # 1. Find untouched closes with early stopping optimization
    # Process from newest to oldest and stop when we have enough levels
    untouched_closes = find_untouched_closes_optimized(
        time_frame, times, closes, highs, lows, opens, volumes, last_Pivot,
        max_above=4, max_below=2
    )

    # Create horizontal lines for untouched closes
    for close_time, close_price, volume in untouched_closes:
        end_time = times[-1]
        # Create horizontal line as two pivots with same price
        start_pivot = (close_time, close_price, volume, None)  # Type doesn't matter for horizontal lines
        end_pivot = (end_time, close_price, volume, None)

        line_type = cast(LineType, f'line_close_{time_frame}')  # Cast string to LineType
        levels_lines.append((start_pivot, end_pivot, line_type, volume))

    # No filtering needed - already optimized during detection
    return {'levels': {'lines': levels_lines}}


def filter_top_levels(levels_lines: List[Line], last_pivot: Pivot, top_above: int = 4, top_below: int = 2) -> List[Line]:
    """
    Filter levels to show only the closest levels relative to last pivot price:
    - 4 closest lines ABOVE the last pivot price
    - 2 closest lines BELOW the last pivot price

    Maximum of 6 lines total (4 above + 2 below)

    Args:
        levels_lines: List of all detected level lines (start_pivot, end_pivot, line_type, volume)
        last_pivot: Last pivot point (timestamp, price, volume, type)
        top_above: Number of closest levels above to include (default: 4)
        top_below: Number of closest levels below to include (default: 2)

    Returns:
        List[Line]: Filtered list of up to 6 levels (4 above, 2 below)
    """
    if not levels_lines:
        return []

    # Extract last pivot price
    last_pivot_price = last_pivot[1]

    # Separate levels into above and below current price
    levels_above = []
    levels_below = []

    for level in levels_lines:
        level_price = level[0][1]  # Get price from start pivot

        if level_price > last_pivot_price:
            levels_above.append(level)
        elif level_price < last_pivot_price:
            levels_below.append(level)
        # Skip levels exactly at current price

    # Sort above levels by price (ascending) - closest first
    levels_above.sort(key=lambda x: x[0][1])

    # Sort below levels by price (descending) - closest first
    levels_below.sort(key=lambda x: x[0][1], reverse=True)

    # Get top N closest from each group
    closest_above = levels_above[:top_above]
    closest_below = levels_below[:top_below]

    # Combine and return
    return closest_below + closest_above


def find_untouched_closes_optimized(
    time_frame: ChartInterval, times, closes, highs, lows, opens, volumes,
    last_pivot: Pivot, max_above: int = 4, max_below: int = 2
) -> List[tuple]:
    """
    Find close prices that were never revisited (untouched).
    OPTIMIZED: Processes candles from newest to oldest and stops early once enough levels found.

    Only detects levels when:
    - There is a color change between consecutive candles
    - AND the close of first candle == open of second candle

    Args:
        time_frame: Chart interval
        times: Array of timestamps
        closes: Array of close prices
        highs: Array of high prices
        lows: Array of low prices
        opens: Array of open prices
        volumes: Array of volumes
        last_pivot: Last pivot point to determine current price
        max_above: Maximum number of levels above current price to find
        max_below: Maximum number of levels below current price to find

    Returns:
        List of (timestamp, price, volume) tuples for untouched closes
    """
    untouched_above = []
    untouched_below = []
    n_candles = len(closes)
    current_price = last_pivot[1]

    # Process candles from newest to oldest (reverse order)
    # Start from n_candles - 2 since we check i+1
    for i in range(n_candles - 2, -1, -1):
        # Early stopping: if we have enough levels on both sides, stop processing
        if len(untouched_above) >= max_above and len(untouched_below) >= max_below:
            break

        close_price = closes[i]
        open_price = opens[i]
        close_time = times[i]
        close_volume = volumes[i]

        # Get next candle data
        next_open = opens[i + 1]
        next_close = closes[i + 1]

        # Check if close of current candle == open of next candle
        if abs(close_price - next_open) < 1e-8:
            # Determine candle colors
            current_is_green = close_price > open_price
            next_is_green = next_close > next_open

            # Only create level if there's a color change
            if current_is_green != next_is_green:
                # Check if this level gets touched by future candles
                is_touched = _check_level_touched_consecutive(
                    i, close_price, highs, lows, n_candles
                )

                if not is_touched:
                    # Separate into above/below based on current price
                    if close_price > current_price and len(untouched_above) < max_above:
                        untouched_above.append((close_time, close_price, close_volume))
                    elif close_price < current_price and len(untouched_below) < max_below:
                        untouched_below.append((close_time, close_price, close_volume))

    # Combine results (already limited by max counts)
    return untouched_below + untouched_above


def find_untouched_closes(time_frame: ChartInterval, times, closes, highs, lows, opens, volumes) -> List[tuple]:
    """
    Find close prices that were never revisited (untouched).
    DEPRECATED: Use find_untouched_closes_optimized instead for better performance.

    Only detects levels when:
    - There is a color change between consecutive candles
    - AND the close of first candle == open of second candle

    Args:
        times: Array of timestamps
        closes: Array of close prices
        highs: Array of high prices
        lows: Array of low prices
        opens: Array of open prices
        volumes: Array of volumes

    Returns:
        List of (timestamp, price, volume) tuples for untouched closes
    """
    untouched = []
    n_candles = len(closes)

    for i in range(n_candles - 1):  # Stop at n_candles - 1 since we check i+1
        close_price = closes[i]
        open_price = opens[i]
        close_time = times[i]
        close_volume = volumes[i]

        # Get next candle data
        next_open = opens[i + 1]
        next_close = closes[i + 1]

        # Check if close of current candle == open of next candle
        if abs(close_price - next_open) < 1e-8:
            # Determine candle colors
            current_is_green = close_price > open_price
            next_is_green = next_close > next_open

            # Only create level if there's a color change
            if current_is_green != next_is_green:
                # Check if this level gets touched by future candles
                is_touched = _check_level_touched_consecutive(
                    i, close_price, highs, lows, n_candles
                )

                if not is_touched:
                    # Validate line type using helper from technical_analysis
                    untouched.append((close_time, close_price, close_volume))
    return untouched


def _check_level_touched_consecutive(candle_idx, close_price, highs, lows, n_candles):
    """
    Helper function to check if a level from consecutive opposite-color candles gets touched.

    Args:
        candle_idx: Index of the first candle in the consecutive pair
        close_price: The price level to test (close of first candle)
        highs: Array of high prices
        lows: Array of low prices
        n_candles: Total number of candles

    Returns:
        bool: True if the level is touched, False if untouched
    """
    # For consecutive candles pattern, check if level gets touched by future candles
    # Skip the next candle (i+1) since it's part of the pattern
    for j in range(candle_idx + 2, n_candles):
        if lows[j] <= close_price <= highs[j]:
            return True
    return False
