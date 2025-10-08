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

    # 1. Find untouched closes (includes special handling for last 2 candles with same close/open)
    untouched_closes = find_untouched_closes(time_frame, times, closes, highs, lows, opens, volumes)

    # Create horizontal lines for untouched closes
    for close_time, close_price, volume in untouched_closes:
        end_time = times[-1]
        # Create horizontal line as two pivots with same price
        start_pivot = (close_time, close_price, None)  # Type doesn't matter for horizontal lines
        end_pivot = (end_time, close_price, None)

        line_type = cast(LineType, f'line_close_{time_frame}')  # Cast string to LineType
        levels_lines.append((start_pivot, end_pivot, line_type, volume))

    # Filter and select top levels based on volume and proximity to last pivot
    filtered_lines = filter_top_levels(levels_lines, last_Pivot)

    return {'levels': {'lines': filtered_lines}}


def filter_top_levels(levels_lines: List[Line], last_pivot: Pivot, top_strong: int = 5, top_close: int = 5) -> List[Line]:
    """
    Filter levels to show only:
    - Top 5 strongest levels (by volume)
    - Top 5 closest levels to last pivot price (from remaining levels)

    Args:
        levels_lines: List of all detected level lines (start_pivot, end_pivot, line_type, volume)
        last_pivot: Last pivot point (timestamp, price, type)
        top_strong: Number of strongest levels to include (default: 5)
        top_close: Number of closest levels to include (default: 5)

    Returns:
        List[Line]: Filtered list of up to 10 levels
    """
    if not levels_lines:
        return []

    # Extract last pivot price once
    last_pivot_price = last_pivot[1]

    # 1. Sort by volume (descending) and get top 5 strongest
    sorted_by_volume = sorted(levels_lines, key=lambda x: x[3] or 0, reverse=True)
    top_5_strongest = sorted_by_volume[:top_strong]

    # 2. For remaining levels, find closest to current price
    # Use simple exclusion by comparing objects directly
    remaining_levels = []
    for level in levels_lines:
        # Check if this level is NOT in top_5_strongest
        is_in_strongest = False
        for strong_level in top_5_strongest:
            if (level[0] == strong_level[0] and  # Same start pivot
                    level[1] == strong_level[1] and  # Same end pivot
                    level[2] == strong_level[2]):    # Same line type
                is_in_strongest = True
                break

        if not is_in_strongest:
            remaining_levels.append(level)

    # Sort remaining by distance to pivot price
    sorted_by_distance = sorted(
        remaining_levels,
        key=lambda x: abs(x[0][1] - last_pivot_price)
    )

    # Get top 5 closest
    top_5_closest = sorted_by_distance[:top_close]

    # Combine and return
    return top_5_strongest + top_5_closest


def find_untouched_closes(time_frame: ChartInterval, times, closes, highs, lows, opens, volumes) -> List[tuple]:
    """
    Find close prices that were never revisited (untouched).

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
