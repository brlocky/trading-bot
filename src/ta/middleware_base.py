"""
Base Middleware Class for Incremental Updates
==============================================

Provides foundation for stateful middlewares that can:
1. Run full calculation (initial state)
2. Update incrementally when new data arrives
3. Cache results to avoid redundant calculations
4. Detect when recalculation is actually needed
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime
from core.trading_types import ChartInterval
from ta.technical_analysis import AnalysisDict, Pivot


@dataclass
class MiddlewareState:
    """
    Stores state for a middleware to enable incremental updates
    """
    # Last timestamp processed
    last_timestamp: pd.Timestamp

    # Last result computed
    last_result: AnalysisDict

    # Number of candles in last computation
    last_candle_count: int

    # Middleware-specific cached data (e.g., pivots, volume bins, etc.)
    cache: Dict[str, Any] = field(default_factory=dict)

    # When this state was created/updated
    updated_at: datetime = field(default_factory=datetime.now)


class IncrementalMiddleware(ABC):
    """
    Base class for middlewares that support incremental updates.

    Subclasses implement:
    - run_full(): Complete calculation from scratch
    - run_incremental(): Update using cached state + new data
    - can_update_incrementally(): Check if incremental update is possible
    """

    def __init__(self):
        # State per timeframe (key: timeframe string)
        self.states: Dict[ChartInterval, MiddlewareState] = {}

    def get_state(self, timeframe: ChartInterval) -> Optional[MiddlewareState]:
        """Get cached state for timeframe"""
        return self.states.get(timeframe)

    def set_state(self, timeframe: ChartInterval, state: MiddlewareState):
        """Cache state for timeframe"""
        self.states[timeframe] = state

    def clear_state(self, timeframe: Optional[ChartInterval] = None):
        """Clear cached state (optionally for specific timeframe)"""
        if timeframe:
            self.states.pop(timeframe, None)
        else:
            self.states.clear()

    @abstractmethod
    def run_full(
        self,
        time_frame: ChartInterval,
        price_data: pd.DataFrame,
        last_pivot: Pivot,
        analysis: AnalysisDict,
        use_log_scale: bool = True,
        **kwargs
    ) -> AnalysisDict:
        """
        Run full calculation from scratch.
        This is the standard middleware calculation.
        """
        pass

    @abstractmethod
    def run_incremental(
        self,
        time_frame: ChartInterval,
        new_candles: pd.DataFrame,
        last_pivot: Pivot,
        analysis: AnalysisDict,
        cached_state: MiddlewareState,
        use_log_scale: bool = True,
        **kwargs
    ) -> AnalysisDict:
        """
        Update calculation using cached state + new candles.

        Args:
            new_candles: Only the NEW data since last calculation
            cached_state: Previous middleware state

        Returns:
            Updated AnalysisDict
        """
        pass

    def can_update_incrementally(
        self,
        time_frame: ChartInterval,
        price_data: pd.DataFrame,
        cached_state: Optional[MiddlewareState]
    ) -> bool:
        """
        Check if incremental update is possible.

        Default implementation checks:
        - State exists
        - New data is strictly newer than cached data
        - Data is continuous (no gaps)
        """
        if cached_state is None:
            return False

        # Check if we have new data
        if price_data.index[-1] <= cached_state.last_timestamp:
            return False  # No new data

        # Check data continuity (no gaps or rewriting history)
        # New data should start after cached data
        new_data_start = price_data.index[cached_state.last_candle_count]
        if new_data_start != cached_state.last_timestamp:
            # Data gap or history rewrite - need full recalc
            return False

        return True

    def run(
        self,
        time_frame: ChartInterval,
        price_data: pd.DataFrame,
        last_pivot: Pivot,
        analysis: AnalysisDict,
        use_log_scale: bool = True,
        force_full: bool = False,
        **kwargs
    ) -> AnalysisDict:
        """
        Smart execution: automatically choose full vs incremental.

        Args:
            force_full: Force full recalculation even if incremental is possible
        """
        cached_state = self.get_state(time_frame)

        # Check if incremental update is possible
        if not force_full and cached_state and self.can_update_incrementally(
            time_frame, price_data, cached_state
        ):
            # Extract only new candles
            new_candles = price_data.iloc[cached_state.last_candle_count:]

            # Run incremental update
            result = self.run_incremental(
                time_frame, new_candles, last_pivot, analysis,
                cached_state, use_log_scale, **kwargs
            )
        else:
            # Run full calculation
            result = self.run_full(
                time_frame, price_data, last_pivot, analysis,
                use_log_scale, **kwargs
            )

        # Cache the new state
        new_state = MiddlewareState(
            last_timestamp=price_data.index[-1],
            last_result=result,
            last_candle_count=len(price_data),
            cache=self._extract_cache_data(result),
        )
        self.set_state(time_frame, new_state)

        return result

    def _extract_cache_data(self, result: AnalysisDict) -> Dict[str, Any]:
        """
        Extract data to cache from result.
        Override in subclasses to cache middleware-specific data.
        """
        return {}


def create_stateful_wrapper(
    middleware_func,
    extract_cache_fn=None,
    can_update_fn=None
):
    """
    Helper to wrap existing middleware functions with incremental update support.

    This allows gradual migration - wrap existing middlewares without rewriting them.

    Args:
        middleware_func: Existing middleware function
        extract_cache_fn: Optional function to extract cacheable data from results
        can_update_fn: Optional custom logic for checking if update is possible
    """

    class WrappedMiddleware(IncrementalMiddleware):
        def run_full(self, time_frame, price_data, last_pivot, analysis, use_log_scale=True, **kwargs):
            return middleware_func(time_frame, price_data, last_pivot, analysis, use_log_scale)

        def run_incremental(self, time_frame, new_candles, last_pivot, analysis,
                            cached_state, use_log_scale=True, **kwargs):
            # Default: fall back to full calculation
            # (Individual middlewares can override this)
            return self.run_full(time_frame, new_candles, last_pivot, analysis, use_log_scale)

        def _extract_cache_data(self, result):
            if extract_cache_fn:
                return extract_cache_fn(result)
            return super()._extract_cache_data(result)

        def can_update_incrementally(self, time_frame, price_data, cached_state):
            if can_update_fn:
                return can_update_fn(time_frame, price_data, cached_state)
            return super().can_update_incrementally(time_frame, price_data, cached_state)

    return WrappedMiddleware()
