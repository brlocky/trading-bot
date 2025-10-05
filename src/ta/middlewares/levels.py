"""
Levels Detection Middleware for Technical Analysis
Detects significant price levels such as support and resistance
"""
import pandas as pd
from typing import List, cast
from core.trading_types import ChartInterval
from src.ta.technical_analysis import Line, AnalysisDict, LineType


def levels_middleware(
    time_frame: ChartInterval,
    price_data: pd.DataFrame,
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

    # 1. Find untouched closes (includes special handling for last 2 candles with same close/open)
    untouched_closes = find_untouched_closes(time_frame, times, closes, highs, lows, opens)

    # 2. Get period high and low lines
    period_lines = get_period_high_low_lines(price_data, time_frame, times)

    # Create horizontal lines for untouched closes
    for close_time, close_price in untouched_closes:
        end_time = times[-1]
        # Create horizontal line as two pivots with same price
        start_pivot = (close_time, close_price, 'high')  # Type doesn't matter for horizontal lines
        end_pivot = (end_time, close_price, 'high')

        line_type = cast(LineType, f'line_close_{time_frame}')  # Cast string to LineType
        levels_lines.append((start_pivot, end_pivot, line_type))

    # Add period high/low lines
    levels_lines.extend(period_lines)

    return {'levels': {'lines': levels_lines}}


def find_untouched_closes(time_frame: ChartInterval, times, closes, highs, lows, opens):
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

    Returns:
        List of (timestamp, price) tuples for untouched closes
    """
    untouched = []
    n_candles = len(closes)

    for i in range(n_candles - 1):  # Stop at n_candles - 1 since we check i+1
        close_price = closes[i]
        open_price = opens[i]
        close_time = times[i]

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
                    untouched.append((close_time, close_price))
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


def get_period_high_low_lines(price_data: pd.DataFrame, time_frame: ChartInterval, times) -> List[Line]:
    """
    Find period high and low levels and return them as horizontal lines.

    Args:
        price_data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        time_frame: Chart interval (e.g., '1h', 'D')
        times: Array of timestamps

    Returns:
        List[Line]: List containing two lines - period high and period low
    """
    lines: List[Line] = []

    # Find period high and low
    period_high_idx = price_data['high'].argmax()
    period_low_idx = price_data['low'].argmin()

    period_high_level = price_data['high'].iloc[period_high_idx]
    period_low_level = price_data['low'].iloc[period_low_idx]

    # Create horizontal lines spanning the entire time range
    start_time = times[0]
    end_time = times[-1]

    # Period high line
    high_start = (start_time, period_high_level, 'high')
    high_end = (end_time, period_high_level, 'high')
    line_type = cast(LineType, f'max_level_{time_frame}')
    lines.append((high_start, high_end, line_type))

    # Period low line
    low_start = (start_time, period_low_level, 'low')
    low_end = (end_time, period_low_level, 'low')
    line_type = cast(LineType, f'min_level_{time_frame}')
    lines.append((low_start, low_end, line_type))

    return lines
