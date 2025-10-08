"""
Zigzag Middleware for Technical Analysis
"""
import numpy as np
import pandas as pd
from typing import List
from core.trading_types import ChartInterval
from ta.technical_analysis import Pivot, Line, AnalysisDict


def zigzag_middleware(
    time_frame: ChartInterval,
    price_data: pd.DataFrame,
    last_Pivot: Pivot,
    analysis: AnalysisDict,
    useLogScale: bool = True,
    threshold: 'float | None' = None,
    factor: float = 0.1
) -> AnalysisDict:
    """
    Robust zigzag: finds true swing highs/lows after threshold reversals
    Returns lines connecting pivots
    """
    highs = price_data['high'].values
    lows = price_data['low'].values
    times = price_data.index
    pivots: List[Pivot] = []
    # Automatically set threshold based on price range if not provided
    if threshold is None:
        price_range = price_data['high'].max() - price_data['low'].min()
        threshold = price_range * factor / price_data['high'].max()
        # threshold is now a relative percent (e.g., 1% of price range relative to max price)
    # Robust initial pivot detection
    initial_high = highs[0]
    initial_low = lows[0]
    initial_idx = 0
    found = False
    start_i = 1
    for i in range(1, len(times)):
        high = highs[i]
        low = lows[i]
        high_broken = high > initial_high
        low_broken = low < initial_low
        if high_broken and not low_broken:
            pivots.append((times[initial_idx], initial_low, 'low'))
            found = True
            start_i = i
            break
        elif low_broken and not high_broken:
            pivots.append((times[initial_idx], initial_high, 'high'))
            found = True
            start_i = i
            break
    if not found:
        pivots.append((times[initial_idx], initial_high, 'high'))
        start_i = 1

    # Continue with zigzag logic
    direction = None
    swing_idx = start_i - 1
    swing_price = highs[swing_idx] if pivots[-1][2] == 'high' else lows[swing_idx]
    swing_type = pivots[-1][2]
    last_pivot_price = swing_price
    for i in range(start_i, len(times)):
        high = highs[i]
        low = lows[i]
        if direction is None:
            change_up = (high - last_pivot_price) / last_pivot_price
            change_down = (low - last_pivot_price) / last_pivot_price
            if abs(change_up) > threshold or abs(change_down) > threshold:
                if abs(change_up) > abs(change_down):
                    direction = 'up'
                    swing_idx = i
                    swing_price = high
                    swing_type = 'high'
                else:
                    direction = 'down'
                    swing_idx = i
                    swing_price = low
                    swing_type = 'low'
        else:
            if direction == 'up':
                if high > swing_price:
                    swing_idx = i
                    swing_price = high
                    swing_type = 'high'
                elif low < swing_price * (1 - threshold):
                    pivots.append((times[swing_idx], swing_price, 'high'))
                    direction = 'down'
                    swing_idx = i
                    swing_price = low
                    swing_type = 'low'
                    last_pivot_price = swing_price
            elif direction == 'down':
                if low < swing_price:
                    swing_idx = i
                    swing_price = low
                    swing_type = 'low'
                elif high > swing_price * (1 + threshold):
                    pivots.append((times[swing_idx], swing_price, 'low'))
                    direction = 'up'
                    swing_idx = i
                    swing_price = high
                    swing_type = 'high'
                    last_pivot_price = swing_price
    # Always add the last swing as a pivot
    if len(pivots) == 0 or pivots[-1][0] != times[-1]:
        pivots.append((times[swing_idx], swing_price, swing_type))

    # Refinement loop (max 5 passes)
    """     max_loops = 5
        loop_count = 0
        while loop_count < max_loops:
            refined_pivots = refine_zigzag(pivots, price_data)
            pivots = refined_pivots
            loop_count += 1
    """
    refine_zigzag(pivots, price_data)

    # Format zigzag lines as pairs of pivots (linear prices only)
    zigzag_lines: List[Line] = [
        ((p1[0], float(p1[1]), p1[2]), (p2[0], float(p2[1]), p2[2]), 'zigzag', None)
        for p1, p2 in zip(pivots[:-1], pivots[1:])
    ]

    return {'zigzag': {'lines': zigzag_lines, 'pivots': pivots}}


def refine_zigzag(
    pivots: list[Pivot],
    price_data: pd.DataFrame,
):
    """
    Refine zigzag pivots by checking if price action between pivots
    violates the zigzag pattern. If so, insert new pivots.
    Returns (refined_pivots, changed)
    """
    i = 0
    while i < len(pivots) - 1:
        p1 = pivots[i]
        p2 = pivots[i+1]
        loc1 = price_data.index.get_loc(p1[0])
        loc2 = price_data.index.get_loc(p2[0])
        p1H = price_data.loc[p1[0], 'high']
        p1L = price_data.loc[p1[0], 'low']
        p2H = price_data.loc[p2[0], 'high']
        p2L = price_data.loc[p2[0], 'low']

        p1_open = price_data.loc[p1[0], 'open']
        p1_close = price_data.loc[p1[0], 'close']
        p2_open = price_data.loc[p2[0], 'open']
        p2_close = price_data.loc[p2[0], 'close']

        if not isinstance(loc1, int) or not isinstance(loc2, int) or not isinstance(p1H, (int, float)) or not isinstance(p1L, (int, float)) or not isinstance(p2H, (int, float)) or not isinstance(p2L, (int, float)) or not isinstance(p1_open, (int, float)) or not isinstance(p1_close, (int, float)) or not isinstance(p2_open, (int, float)) or not isinstance(p2_close, (int, float)):
            i += 1
            print("Non-numeric data found, skipping this pair of pivots.")
            continue

        idx1 = loc1
        idx2 = loc2

        p1color = 'red' if p1_open > p1_close else 'green'
        p2color = 'red' if p2_open > p2_close else 'green'

        if idx2 <= idx1 + 1:
            if (p1color == 'red' and p1[2] == 'high' and p2[2] == 'low' and p1L < p2L):
                pivots[i+1] = (p1[0], float(p1L), 'low')
                i += 1
                continue

            if (p1color == 'green' and p1[2] == 'low' and p2[2] == 'high' and p1H > p2H):
                pivots[i+1] = (p1[0], float(p1H), 'high')
                i += 1
                continue

            if p1[2] == p2[2]:
                print("Refined zigzag pivots due to same type violation.", p1[0], p2[0])

            i += 1
            continue

        segment = price_data.iloc[idx1+1:idx2]
        highs = segment['high'].to_numpy()
        lows = segment['low'].to_numpy()

        # Type cast to handle potential mypy confusion about column types
        max_high: float = float(np.max(highs.astype(float)))
        max_idx = int(np.argmax(highs))
        max_abs_idx = segment.index[int(max_idx)]

        min_low: float = float(np.min(lows.astype(float)))
        min_idx = int(np.argmin(lows))
        min_abs_idx = segment.index[int(min_idx)]

        # Check for violations between p1 and p2
        # Same type pivots: must have opposite extreme in between
        # Case 1: Two consecutive lows, insert max high between them
        if p1[2] == 'low' and p2[2] == 'low':
            pivots.insert(i+1, (max_abs_idx, max_high, 'high'))
            continue

        # Case 2: Two consecutive highs, insert min low between them
        if p1[2] == 'high' and p2[2] == 'high':
            pivots.insert(i+1, (min_abs_idx, min_low, 'low'))
            continue

        # Case 3: p2 is a low, but a lower low exists in the segment
        if min_low < p2[1] and p2[2] == 'low':
            pivots[i+1] = (min_abs_idx, min_low, 'low')
            continue

        # Case 4: p2 is a high, but a higher high exists in the segment
        if max_high > p2[1] and p2[2] == 'high':
            pivots[i+1] = (max_abs_idx, max_high, 'high')
            continue

        # Case 5: p1 is a low, but a lower low exists in the segment
        if min_low < p1[1] and p1[2] == 'low':
            pivots[i] = (min_abs_idx, min_low, 'low')
            continue

        # Case 6: p1 is a high, but a higher high exists in the segment
        if max_high > p1[1] and p1[2] == 'high':
            pivots[i] = (max_abs_idx, max_high, 'high')
            continue

        # Case 7: If p2 is a red candle, is a low, but its high is greater than previous high pivot
        # Update previous high pivot to use p2's timestamp and high value
        if p2color == 'red' and p1[2] == 'high' and p2[2] == 'low' and p2H > p1H:
            pivots[i] = (p2[0], float(p2H), 'high')
            continue

        # Case 8: If p2 is a green candle, is a high, but its low is lower than previous low pivot
        # Update previous low pivot to use p2's timestamp and low value
        if p2color == 'green' and p1[2] == 'low' and p2[2] == 'high' and p2L < p1L:
            pivots[i] = (p2[0], float(p2L), 'low')
            continue

        # Add more cases here for other violations
        i += 1
