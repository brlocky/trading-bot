
"""
Volume Profile Middleware for Technical Analysis
Returns AnalysisDict compatible with other middlewares.
"""
from typing import List
import pandas as pd
from core.trading_types import ChartInterval
from src.ta.technical_analysis import AnalysisDict, Line
from src.ta.middlewares.global_volume_profile import calculate_volume_profile_histogram

# Number of last periods to process for volume profile calculation


def volume_profile_middleware(
    interval: ChartInterval,
    price_data: pd.DataFrame,
    analysis: AnalysisDict,
    useLogScale: bool = True,
    bins: int = 255,
) -> AnalysisDict:
    """
    Calculates volume profile as a histogram of volume at price bins.
    Returns AnalysisDict with 'volume_profile_periods' key.
    Args:
        price_data: DataFrame with 'close' and 'volume' columns.
        analysis: AnalysisDict from previous middlewares.
        useLogScale: Whether to use log scale for prices.
        bins: Number of bins.
        interval: Optional interval string from JSON metadata (e.g., '1d', '1w', '1M').
    Returns:
        AnalysisDict: {'volume_profile_periods': {'vp': [VolumeProfilePeriod, ...]}}
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

    # Helper to get period ranges using pointers
    def get_period_ranges(df, range_type):
        periods = []
        start_date = df.index.min()
        end_date = df.index.max()

        if range_type == 'day':
            # Daily periods
            current = start_date.normalize()
            while current <= end_date:
                next_day = current + pd.Timedelta(days=1)
                start_idx = df.index.searchsorted(current, side='left')
                end_idx = df.index.searchsorted(next_day, side='left')
                if start_idx < end_idx:
                    periods.append((current.strftime('%Y-%m-%d'), start_idx, end_idx))
                current = next_day

        elif range_type == 'week':
            # Weekly periods
            current = start_date - pd.Timedelta(days=start_date.weekday())
            while current <= end_date:
                next_week = current + pd.Timedelta(weeks=1)
                start_idx = df.index.searchsorted(current, side='left')
                end_idx = df.index.searchsorted(next_week, side='left')
                if start_idx < end_idx:
                    periods.append((current.strftime('%Y-W%U'), start_idx, end_idx))
                current = next_week

        elif range_type == 'month':
            # Monthly periods
            current = start_date.replace(day=1)
            while current <= end_date:
                if current.month == 12:
                    next_month = current.replace(year=current.year + 1, month=1)
                else:
                    next_month = current.replace(month=current.month + 1)
                start_idx = df.index.searchsorted(current, side='left')
                end_idx = df.index.searchsorted(next_month, side='left')
                if start_idx < end_idx:
                    periods.append((current.strftime('%Y-%m'), start_idx, end_idx))
                current = next_month

        elif range_type == 'year':
            # Yearly periods
            current = start_date.replace(month=1, day=1)
            while current <= end_date:
                next_year = current.replace(year=current.year + 1)
                start_idx = df.index.searchsorted(current, side='left')
                end_idx = df.index.searchsorted(next_year, side='left')
                if start_idx < end_idx:
                    periods.append((current.strftime('%Y'), start_idx, end_idx))
                current = next_year
        else:
            # Full range
            periods = [('full', 0, len(df))]

        return periods

    vp_ranges: List[Line] = []
    poc_levels = []  # Store POC levels for efficient O(n) naked testing

    periods = get_period_ranges(price_data, range_type)
    for period_name, start_idx, end_idx in periods:
        # Use pointers to slice data without copying
        period_data = price_data.iloc[start_idx:end_idx]
        prices = period_data['close'].to_numpy(dtype=float)
        volumes = period_data['volume'].to_numpy(dtype=float)

        if prices.size == 0 or volumes.size == 0:
            continue

        # Use shared volume profile calculation function
        try:
            bin_edges, bin_volumes, min_price, max_price, key_levels = calculate_volume_profile_histogram(
                prices, volumes, bins
            )
        except ValueError:
            # Skip periods with insufficient data (handled by shared function)
            continue

        # Extract key levels from shared calculation
        poc_price_mid = key_levels['poc_price']
        poc_volume = key_levels['poc_volume']
        vah_price = key_levels['vah_price']
        val_price = key_levels['val_price']

        # Get period timing
        start_time: pd.Timestamp | None = period_data.index[0] if len(period_data.index) > 0 else None
        period_end_time: pd.Timestamp | None = period_data.index[-1] if len(period_data.index) > 0 else None

        if start_time is None or period_end_time is None:
            continue

        # Skip period if we can't calculate proper levels (insufficient data)
        total_volume = sum(bin_volumes)
        if len(bin_volumes) < 3 or total_volume == 0:
            print(f"  Period {period_name}: SKIPPED - insufficient data")
            continue

        # Store POC for later O(n) naked testing
        poc_levels.append({
            'period_name': period_name,
            'start_time': start_time,
            'period_end_time': period_end_time,
            'poc_price': poc_price_mid,
            'poc_volume': poc_volume
        })

        # Add VAH and VAL immediately (always end at period end)
        vp_ranges.append((
            (start_time, vah_price, None), (period_end_time, vah_price, None), 'vah_line'
        ))

        vp_ranges.append((
            (start_time, val_price, None), (period_end_time, val_price, None), 'val_line'
        ))

    # O(n) naked POC detection - single pass through price data
    # Sort POCs by period end time for efficient processing
    poc_levels.sort(key=lambda x: x['period_end_time'])

    # Process all POCs in one pass through price data
    active_pocs = []  # POCs that are still being tested for touches

    for timestamp in price_data.index:
        candle = price_data.loc[timestamp]
        candle_low = candle['low']
        candle_high = candle['high']

        # Add any POCs that became active at this timestamp
        while poc_levels and timestamp >= poc_levels[0]['period_end_time']:
            poc_info = poc_levels.pop(0)
            poc_info['end_time'] = price_data.index[-1]  # Default to end of data
            active_pocs.append(poc_info)

        # Check active POCs for touches
        remaining_pocs = []
        for poc_info in active_pocs:
            if candle_low <= poc_info['poc_price'] <= candle_high:
                # POC was touched, set end time
                poc_info['end_time'] = timestamp
                # Add to final ranges
                vp_ranges.append((
                    (poc_info['start_time'], poc_info['poc_price'], None), (timestamp, poc_info['poc_price'], None), 'poc_line'
                ))
            else:
                # POC not touched yet, keep testing
                remaining_pocs.append(poc_info)

        active_pocs = remaining_pocs

    # Add remaining untouched POCs (naked levels that extend to end)
    for poc_info in active_pocs:
        vp_ranges.append((
            (poc_info['start_time'], poc_info['poc_price'], None), (poc_info['end_time'], poc_info['poc_price'], None), 'naked_poc_line'
        ))

    print(f"Volume Profile: Generated {len(vp_ranges)} volume profile ranges")
    return {'volume_profile_periods': {'lines': vp_ranges}}
