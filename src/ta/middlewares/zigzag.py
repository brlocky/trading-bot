"""
Zigzag Middleware for Technical Analysis
"""
import numpy as np
import pandas as pd
from typing import List, Literal, cast
from core.trading_types import ChartInterval
from ta.technical_analysis import Pivot, Line, AnalysisDict


def classify_pivot_swing_type(pivots: List[Pivot]) -> List[Pivot]:
    """
    Classify pivots as HH (Higher High), LL (Lower Low), HL (Higher Low), or LH (Lower High).

    Logic:
    - Compare each high pivot with the previous high pivot
    - Compare each low pivot with the previous low pivot
    - HH: Current high > Previous high
    - LH: Current high < Previous high
    - HL: Current low > Previous low
    - LL: Current low < Previous low

    Args:
        pivots: List of pivots with None as the swing type

    Returns:
        List of pivots with classified swing types (HH, LL, HL, LH)
    """
    if len(pivots) < 2:
        return pivots

    classified_pivots: List[Pivot] = []
    last_high_price: float | None = None
    last_low_price: float | None = None

    for i, pivot in enumerate(pivots):
        timestamp, price, volume, _ = pivot

        # Determine if this is a high or low pivot by checking neighbors
        is_high = False
        is_low = False

        # Check by comparing with adjacent pivots
        if i == 0:
            # First pivot - check next
            if i + 1 < len(pivots):
                next_price = pivots[i + 1][1]
                is_high = price > next_price
                is_low = price < next_price
        elif i == len(pivots) - 1:
            # Last pivot - check previous
            prev_price = pivots[i - 1][1]
            is_high = price > prev_price
            is_low = price < prev_price
        else:
            # Middle pivot - check both neighbors
            prev_price = pivots[i - 1][1]
            next_price = pivots[i + 1][1]
            is_high = price > prev_price and price > next_price
            is_low = price < prev_price and price < next_price

        swing_type: Literal['HH', 'HL', 'LL', 'LH'] | None = None

        if is_high:
            if last_high_price is not None:
                swing_type = 'HH' if price > last_high_price else 'LH'
            else:
                swing_type = 'HH'  # First high defaults to HH
            last_high_price = price
        elif is_low:
            if last_low_price is not None:
                swing_type = 'LL' if price < last_low_price else 'HL'
            else:
                swing_type = 'LL'  # First low defaults to LL
            last_low_price = price

        classified_pivots.append((timestamp, price, volume, swing_type))

    return classified_pivots


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
    Returns lines connecting pivots with HH/LL/HL/LH classifications
    """
    highs = price_data['high'].values
    lows = price_data['low'].values
    volumes = price_data['volume'].values
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
    first_pivot_is_high = True  # Track whether first pivot was a high or low

    for i in range(1, len(times)):
        high = highs[i]
        low = lows[i]
        high_broken = high > initial_high
        low_broken = low < initial_low
        if high_broken and not low_broken:
            # Price went up first, so first pivot is a LOW
            pivots.append((times[initial_idx], initial_low, volumes[initial_idx], None))
            first_pivot_is_high = False
            found = True
            start_i = i
            break
        elif low_broken and not high_broken:
            # Price went down first, so first pivot is a HIGH
            pivots.append((times[initial_idx], initial_high, volumes[initial_idx], None))
            first_pivot_is_high = True
            found = True
            start_i = i
            break
    if not found:
        # No clear direction, start with high
        pivots.append((times[initial_idx], initial_high, volumes[initial_idx], None))
        first_pivot_is_high = True
        start_i = 1

    # Continue with zigzag logic
    direction = None
    swing_idx = start_i - 1

    # Set swing_price based on whether first pivot was high or low
    swing_price = highs[swing_idx] if first_pivot_is_high else lows[swing_idx]
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
                    pivots.append((times[swing_idx], swing_price, volumes[swing_idx], None))
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
                    pivots.append((times[swing_idx], swing_price, volumes[swing_idx], None))
                    direction = 'up'
                    swing_idx = i
                    swing_price = high
                    swing_type = 'high'
                    last_pivot_price = swing_price
    # Always add the last swing as a pivot
    if len(pivots) == 0 or pivots[-1][0] != times[-1]:
        pivots.append((times[swing_idx], swing_price, volumes[swing_idx], None))

    # Refinement loop (max 5 passes)
    """     max_loops = 5
        loop_count = 0
        while loop_count < max_loops:
            refined_pivots = refine_zigzag(pivots, price_data)
            pivots = refined_pivots
            loop_count += 1
    """
    refine_zigzag(pivots, price_data)

    # ⚡ CLASSIFY PIVOTS: Determine HH, LL, HL, LH swing types
    pivots = classify_pivot_swing_type(pivots)

    # Format zigzag lines as pairs of pivots (linear prices only)
    zigzag_lines: List[Line] = [
        ((p1[0], float(p1[1]), p1[2], p1[3]), (p2[0], float(p2[1]), p2[2], p2[3]), 'zigzag', None)
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

    Note: At this stage, pivots have None as type (will be classified later).
    We determine if a pivot is a high or low by comparing with neighbors.
    """
    i = 0
    while i < len(pivots) - 1:
        p1 = pivots[i]
        p2 = pivots[i+1]
        loc1 = cast(int, price_data.index.get_loc(p1[0]))
        loc2 = cast(int, price_data.index.get_loc(p2[0]))
        p1H = cast(float, price_data.at[p1[0], 'high'])
        p1L = cast(float, price_data.loc[p1[0], 'low'])
        p2H = cast(float, price_data.loc[p2[0], 'high'])
        p2L = cast(float, price_data.loc[p2[0], 'low'])

        p1_open = cast(float, price_data.loc[p1[0], 'open'])
        p1_close = cast(float, price_data.loc[p1[0], 'close'])
        p2_open = cast(float, price_data.loc[p2[0], 'open'])
        p2_close = cast(float, price_data.loc[p2[0], 'close'])
        p1_volume = cast(float, price_data.loc[p1[0], 'volume'])
        p2_volume = cast(float, price_data.loc[p2[0], 'volume'])

        idx1 = loc1
        idx2 = loc2

        p1color = 'red' if p1_open > p1_close else 'green'
        p2color = 'red' if p2_open > p2_close else 'green'

        # Determine if pivots are highs or lows by comparing prices
        p1_is_high = p1[1] > p2[1]
        p1_is_low = p1[1] < p2[1]
        p2_is_high = p2[1] > p1[1]
        p2_is_low = p2[1] < p1[1]

        if idx2 <= idx1 + 1:
            if (p1color == 'red' and p1_is_high and p2_is_low and p1L < p2L):
                pivots[i+1] = (p1[0], p1L, p1_volume, None)
                i += 1
                continue

            if (p1color == 'green' and p1_is_low and p2_is_high and p1H > p2H):
                pivots[i+1] = (p1[0], p1H, p1_volume, None)
                i += 1
                continue

            # Check if consecutive pivots have same relationship (both higher or both lower)
            if not (p1_is_high or p1_is_low):
                print("Refined zigzag pivots due to same price violation.", p1[0], p2[0])

            i += 1
            continue

        segment = price_data.iloc[idx1+1:idx2]
        highs = segment['high'].to_numpy()
        lows = segment['low'].to_numpy()
        volumes_seg = segment['volume'].to_numpy()

        # Type cast to handle potential mypy confusion about column types
        max_high: float = float(np.max(highs.astype(float)))
        max_idx = int(np.argmax(highs))
        max_abs_idx = segment.index[int(max_idx)]
        max_volume = float(volumes_seg[max_idx])

        min_low: float = float(np.min(lows.astype(float)))
        min_idx = int(np.argmin(lows))
        min_abs_idx = segment.index[int(min_idx)]
        min_volume = float(volumes_seg[min_idx])

        # Check for violations between p1 and p2
        # Same type pivots: must have opposite extreme in between
        # Case 1: Two consecutive lows (both prices lower than next), insert max high between them
        if p1_is_low and p2_is_low:
            pivots.insert(i+1, (max_abs_idx, max_high, max_volume, None))
            continue

        # Case 2: Two consecutive highs (both prices higher than next), insert min low between them
        if p1_is_high and p2_is_high:
            pivots.insert(i+1, (min_abs_idx, min_low, min_volume, None))
            continue

        # Case 3: p2 is a low, but a lower low exists in the segment
        if min_low < p2[1] and p2_is_low:
            pivots[i+1] = (min_abs_idx, min_low, min_volume, None)
            continue

        # Case 4: p2 is a high, but a higher high exists in the segment
        if max_high > p2[1] and p2_is_high:
            pivots[i+1] = (max_abs_idx, max_high, max_volume, None)
            continue

        # Case 5: p1 is a low, but a lower low exists in the segment
        if min_low < p1[1] and p1_is_low:
            pivots[i] = (min_abs_idx, min_low, min_volume, None)
            continue

        # Case 6: p1 is a high, but a higher high exists in the segment
        if max_high > p1[1] and p1_is_high:
            pivots[i] = (max_abs_idx, max_high, max_volume, None)
            continue

        # Case 7: If p2 is a red candle, is a low, but its high is greater than previous high pivot
        # Update previous high pivot to use p2's timestamp and high value
        if p2color == 'red' and p1_is_high and p2_is_low and p2H > p1H:
            pivots[i] = (p2[0], float(p2H), float(p2_volume), None)
            continue

        # Case 8: If p2 is a green candle, is a high, but its low is lower than previous low pivot
        # Update previous low pivot to use p2's timestamp and low value
        if p2color == 'green' and p1_is_low and p2_is_high and p2L < p1L:
            pivots[i] = (p2[0], float(p2L), float(p2_volume), None)
            continue

        # Add more cases here for other violations
        i += 1
