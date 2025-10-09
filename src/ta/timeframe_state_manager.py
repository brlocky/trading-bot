"""
Timeframe State Manager
=======================

Manages state and caching for multi-timeframe TA calculations.
Tracks when each timeframe actually changes to avoid redundant recalculation.
"""

from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from core.trading_types import ChartInterval


@dataclass
class TimeframeState:
    """
    Tracks state for a single timeframe to detect when recalculation is needed.
    """
    # First candle timestamp for this timeframe
    first_timestamp: pd.Timestamp

    # Last candle timestamp for this timeframe
    last_timestamp: pd.Timestamp

    # Number of candles in last calculation
    candle_count: int

    # Cached TA results for this timeframe (can be Dict or List)
    cached_result: Dict = field(default_factory=dict)

    # Hash of data to detect changes
    data_hash: Optional[str] = None

    # When this was last updated
    updated_at: datetime = field(default_factory=datetime.now)

    # Configuration parameters that affect calculation results
    use_log_scale: bool = True


class TimeframeStateManager:
    """
    Manages state across all timeframes to intelligently trigger recalculations.

    Key optimization: Only recalculate a timeframe when it actually has new data.
    For example, when processing a new 15m candle:
    - 15m timeframe: Always recalculate (new candle)
    - 1h timeframe: Only recalculate when hour completes (every 4th 15m candle)
    - Daily: Only when day completes (every 96th 15m candle)
    - etc.
    """

    def __init__(self):
        self.states: Dict[ChartInterval, TimeframeState] = {}

    def needs_recalculation(
        self,
        timeframe: ChartInterval,
        current_data: pd.DataFrame,
        force: bool = False,
        use_log_scale: bool = True
    ) -> bool:
        """
        Determine if a timeframe needs recalculation.

        Returns True if:
        - No cached state exists (first run)
        - Last candle timestamp changed (new data)
        - Log scale parameter changed
        - Force flag is set

        Returns False if:
        - Last candle timestamp is identical (no new data)
        - Log scale parameter is unchanged

        NOTE: We only check the last timestamp, not length, because
        the data might be filtered by lookback windows (e.g., 31 days for 1h)
        which changes length even when no new candles arrive.
        """
        if force:
            return True

        state = self.states.get(timeframe)
        if state is None:
            return True  # First run

        if len(current_data) == 0:
            return False  # No data to process

        # Check if log_scale parameter changed
        if state.use_log_scale != use_log_scale:
            return True  # Configuration changed - recalculate!

        # Check if data range changed (start or end date)
        if current_data.index[0] != state.first_timestamp:
            return True  # Start date changed - different data range!

        if current_data.index[-1] != state.last_timestamp:
            return True  # End date changed - new data!

        # Check if number of candles changed (data might be filtered differently)
        if len(current_data) != state.candle_count:
            return True  # Candle count changed - recalculate!

        return False  # No changes detected

    def _hash_recent_data(self, data: pd.DataFrame, last_n: int = 1) -> str:
        """
        Create simple identifier for last candle.
        Used for tracking state changes.
        """
        if len(data) == 0:
            return "empty"
        # Simple identifier: last timestamp + last close price
        hash_str = f"{data.index[-1]}_{data['close'].iloc[-1]:.2f}"
        return hash_str

    def update_state(
        self,
        timeframe: ChartInterval,
        data: pd.DataFrame,
        result: Dict,
        use_log_scale: bool = True
    ):
        """Update cached state for a timeframe"""
        state = TimeframeState(
            first_timestamp=data.index[0],
            last_timestamp=data.index[-1],
            candle_count=len(data),
            cached_result=result,
            data_hash=self._hash_recent_data(data),
            updated_at=datetime.now(),
            use_log_scale=use_log_scale
        )
        self.states[timeframe] = state

    def get_cached_result(
        self,
        timeframe: ChartInterval
    ) -> Optional[Dict]:
        """Get cached result for a timeframe"""
        state = self.states.get(timeframe)
        return state.cached_result if state else None

    def clear_state(self, timeframe: Optional[ChartInterval] = None):
        """Clear cached state"""
        if timeframe:
            self.states.pop(timeframe, None)
        else:
            self.states.clear()

    def get_timeframes_needing_update(
        self,
        data_dfs: Dict[ChartInterval, pd.DataFrame],
        force_timeframes: Optional[Set[ChartInterval]] = None,
        use_log_scale: bool = True
    ) -> Set[ChartInterval]:
        """
        Get set of timeframes that need recalculation.

        Args:
            data_dfs: Current data for all timeframes
            force_timeframes: Timeframes to force update regardless of state
            use_log_scale: Whether log scale is being used (affects caching)

        Returns:
            Set of timeframes that need processing
        """
        needs_update = set()

        for timeframe, data in data_dfs.items():
            should_force = bool(force_timeframes and timeframe in force_timeframes)
            if self.needs_recalculation(timeframe, data, force=should_force, use_log_scale=use_log_scale):
                needs_update.add(timeframe)

        return needs_update

    def get_stats(self) -> Dict:
        """Get statistics about cached states"""
        return {
            'cached_timeframes': list(self.states.keys()),
            'total_cached': len(self.states),
            'cache_ages': {
                tf: (datetime.now() - state.updated_at).total_seconds()
                for tf, state in self.states.items()
            }
        }
