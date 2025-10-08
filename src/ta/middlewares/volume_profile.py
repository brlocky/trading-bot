
"""
Volume Profile Middleware for Technical Analysis - OPTIMIZED VERSION
Returns AnalysisDict compatible with other middlewares.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from core.trading_types import ChartInterval
from ta.technical_analysis import AnalysisDict, Line, Pivot
from ta.middlewares.global_volume_profile import calculate_key_volume_levels_fast


def get_period_ranges_fast(df: pd.DataFrame, range_type: str) -> List[tuple]:
    """OPTIMIZED: Fast period range calculation using vectorized operations."""
    periods = []

    if range_type == 'day':
        # Group by date only
        dates = pd.to_datetime(df.index).normalize()
        unique_dates = dates.unique()
        for date in unique_dates:
            mask = dates == date
            indices = np.where(mask)[0]
            if len(indices) > 0:
                periods.append((date.strftime('%Y-%m-%d'), indices[0], indices[-1] + 1))

    elif range_type == 'week':
        # Group by week
        weeks = pd.to_datetime(df.index).to_period('W')
        unique_weeks = weeks.unique()
        for week in unique_weeks:
            mask = weeks == week
            indices = np.where(mask)[0]
            if len(indices) > 0:
                periods.append((str(week), indices[0], indices[-1] + 1))

    elif range_type == 'month':
        # Group by month
        months = pd.to_datetime(df.index).to_period('M')
        unique_months = months.unique()
        for month in unique_months:
            mask = months == month
            indices = np.where(mask)[0]
            if len(indices) > 0:
                periods.append((str(month), indices[0], indices[-1] + 1))

    elif range_type == 'year':
        # Group by year
        years = pd.to_datetime(df.index).to_period('Y')
        unique_years = years.unique()
        for year in unique_years:
            mask = years == year
            indices = np.where(mask)[0]
            if len(indices) > 0:
                periods.append((str(year), indices[0], indices[-1] + 1))
    else:
        # Full range
        periods = [('full', 0, len(df))]

    return periods


def detect_naked_pocs_fast(price_data: pd.DataFrame, poc_levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OPTIMIZED: Vectorized naked POC detection using numpy operations."""
    if not poc_levels:
        return []

    # Pre-extract arrays for vectorized operations
    highs = price_data['high'].values
    lows = price_data['low'].values

    # Process each POC
    for poc_info in poc_levels:
        poc_price = poc_info['poc_price']
        period_end_idx = price_data.index.get_indexer([poc_info['period_end_time']])[0]

        # Check only candles after period end
        if period_end_idx + 1 >= len(price_data):
            # POC at end of data - remains naked
            poc_info['end_time'] = price_data.index[-1]
            poc_info['is_naked'] = True
            continue

        # Vectorized check for touches after period end
        future_highs = highs[period_end_idx + 1:]
        future_lows = lows[period_end_idx + 1:]

        # Check if POC price is within any future candle's range
        touches = (future_lows <= poc_price) & (poc_price <= future_highs)

        if np.any(touches):
            # Find first touch - use pandas index to maintain Timestamp type
            touch_idx = np.where(touches)[0][0] + period_end_idx + 1
            poc_info['end_time'] = price_data.index[touch_idx]
            poc_info['is_naked'] = False
        else:
            # No touches found - remains naked
            poc_info['end_time'] = price_data.index[-1]
            poc_info['is_naked'] = True

    return poc_levels


def volume_profile_middleware(
    interval: ChartInterval,
    price_data: pd.DataFrame,
    last_Pivot: Pivot,
    analysis: AnalysisDict,
    useLogScale: bool = True,
    bins: int = 100,
    keep_last_n_sessions: int | None = 5,
) -> AnalysisDict:
    """
    OPTIMIZED: Fast volume profile calculation focusing only on POC/VAH/VAL.
    Includes efficient naked POC detection using vectorized operations.
    """
    # Determine range type based on interval
    if interval in ['D']:
        range_type = 'week'
    elif interval in ['W']:
        range_type = 'month'
    elif interval in ['M']:
        range_type = 'year'
    else:
        range_type = 'day'

    vp_ranges: List[Line] = []
    poc_levels = []

    # Use optimized period calculation
    periods = get_period_ranges_fast(price_data, range_type)

    for period_name, start_idx, end_idx in periods:
        # Direct array slicing - much faster than DataFrame operations
        period_prices = price_data['close'].iloc[start_idx:end_idx].to_numpy(dtype=float)
        period_volumes = price_data['volume'].iloc[start_idx:end_idx].to_numpy(dtype=float)

        if len(period_prices) == 0:
            continue

        # Use optimized key levels calculation
        try:
            key_levels = calculate_key_volume_levels_fast(period_prices, period_volumes, bins)
        except ValueError:
            continue

        # Get period timing - use native indexing
        period_start_time = price_data.index[start_idx]  # type: ignore
        period_end_time = price_data.index[end_idx - 1]  # type: ignore

        # Store POC for naked testing
        poc_levels.append({
            'period_name': period_name,
            'start_time': period_start_time,
            'period_end_time': period_end_time,
            'poc_price': key_levels['poc_price'],
            'poc_volume': key_levels['poc_volume']
        })

        # Add VAH and VAL with proper 4-tuple format (Line type)
        vp_ranges.append((
            (period_start_time, key_levels['vah_price'], None),
            (period_end_time, key_levels['vah_price'], None),
            'vah_line',
            int(key_levels['vah_volume'])
        ))

        vp_ranges.append((
            (period_start_time, key_levels['val_price'], None),
            (period_end_time, key_levels['val_price'], None),
            'val_line',
            int(key_levels['val_volume'])
        ))

    # Fast naked POC detection
    poc_levels = detect_naked_pocs_fast(price_data, poc_levels)

    # Filter VAH/VAL based on keep_last_n_sessions
    if keep_last_n_sessions is not None:
        # Group VAH/VAL by session
        sessions: Dict[pd.Timestamp, List[int]] = {}
        for idx, line in enumerate(vp_ranges):
            if 'vah_line' in line[2] or 'val_line' in line[2]:
                session_start = line[0][0]
                if session_start not in sessions:
                    sessions[session_start] = []
                sessions[session_start].append(idx)

        # Keep only last N sessions
        recent_sessions = sorted(sessions.keys())[-keep_last_n_sessions:]
        indices_to_keep = set()
        for session_start in recent_sessions:
            indices_to_keep.update(sessions[session_start])

        # Filter vp_ranges
        vp_ranges = [line for idx, line in enumerate(vp_ranges) if idx in indices_to_keep]

    # Add POCs based on filtering rules
    recent_session_starts = set()
    if keep_last_n_sessions is not None:
        all_sessions = sorted(set(poc['start_time'] for poc in poc_levels))
        recent_session_starts = set(all_sessions[-keep_last_n_sessions:])

    for poc_info in poc_levels:
        if poc_info['is_naked']:
            # Keep ALL naked POCs
            vp_ranges.append((
                (poc_info['start_time'], poc_info['poc_price'], None),
                (poc_info['end_time'], poc_info['poc_price'], None),
                'naked_poc_line',
                int(poc_info['poc_volume'])
            ))
        elif poc_info['start_time'] in recent_session_starts:
            # Keep touched POCs only from recent sessions
            vp_ranges.append((
                (poc_info['start_time'], poc_info['poc_price'], None),
                (poc_info['end_time'], poc_info['poc_price'], None),
                'poc_line',
                int(poc_info['poc_volume'])
            ))

    return {'volume_profile_periods': {'lines': vp_ranges}}
