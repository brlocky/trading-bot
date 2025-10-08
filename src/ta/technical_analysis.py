"""
Technical Analysis Processor
Supports middleware for candlestick data analysis (e.g., zigzag)

This module provides a class for running technical analysis on candlestick data using a middleware pattern.
Each middleware can process the data and add its own analysis results (such as pivots, lines, volume profiles).
"""


# Import necessary libraries
import pandas as pd
from typing import List, Callable, Dict, Optional, Tuple, Literal
from core.trading_types import ChartInterval
from typing import TypedDict


# Type definitions for analysis results
# Pivot: (timestamp, price, type)
Pivot = Tuple[pd.Timestamp, float, Optional[Literal['high', 'low']]]


LineType = Literal['zigzag',
                   'vah_line', 'val_line', 'poc_line', 'naked_poc_line',
                   'channel_upper_line', 'channel_lower_line', 'channel_middle_line', 'channel_inner_line',
                   # Lines and levels for all intervals
                   'line_close_1m', 'max_level_1m', 'min_level_1m',
                   'line_close_2m', 'max_level_2m', 'min_level_2m',
                   'line_close_3m', 'max_level_3m', 'min_level_3m',
                   'line_close_5m', 'max_level_5m', 'min_level_5m',
                   'line_close_10m', 'max_level_10m', 'min_level_10m',
                   'line_close_15m', 'max_level_15m', 'min_level_15m',
                   'line_close_30m', 'max_level_30m', 'min_level_30m',
                   'line_close_1h', 'max_level_1h', 'min_level_1h',
                   'line_close_2h', 'max_level_2h', 'min_level_2h',
                   'line_close_4h', 'max_level_4h', 'min_level_4h',
                   'line_close_D', 'max_level_D', 'min_level_D',
                   'line_close_W', 'max_level_W', 'min_level_W',
                   'line_close_M', 'max_level_M', 'min_level_M'
                   ]

# Line: (pivot1, pivot2, line_type, volume)
Line = Tuple[Pivot, Pivot, LineType, Optional[int]]  # Added int for strength or other metadata

# VolumeProfileLine: ((bin_start, bin_end), bin_volume, normalized_volume)
VolumeProfileLine = Tuple[Tuple[float, float], float, float]

# MiddlewareResult: dictionary structure for middleware output


class MiddlewareResult(TypedDict, total=False):
    lines: List[Line]  # List of lines (e.g., trend lines)
    pivots: List[Pivot]  # List of pivot points
    vp: List[VolumeProfileLine]  # List of volume profile bins


# AnalysisDict: stores results from all middlewares
AnalysisDict = Dict[str, MiddlewareResult]


class TechnicalAnalysisProcessor:

    """
    Main processor for running technical analysis on candlestick data.
    Uses a middleware pattern to allow extensible analysis modules.
    """

    def __init__(self, df: pd.DataFrame, time_frame: ChartInterval, last_pivot: Pivot, useLogScale: bool = True):
        """
        Initialize the processor with a pandas DataFrame containing candlestick data.
        Args:
            df (pd.DataFrame): Candlestick data (OHLCV)
        """
        # Store the DataFrame
        self.df = df
        self.time_frame: ChartInterval = time_frame
        # Whether to use logarithmic scale for price-based analyses
        self.useLogScale = useLogScale
        # List of middleware functions to apply
        self.middlewares: List[Callable[[ChartInterval, pd.DataFrame,  Pivot, AnalysisDict, bool], AnalysisDict]] = []
        # Store the last pivot
        self.last_pivot = last_pivot

    def register_middleware(self, middleware: Callable[[ChartInterval, pd.DataFrame, Pivot, AnalysisDict, bool], AnalysisDict]):
        """
        Register a middleware function for analysis.
        Args:
            middleware (Callable): Function that takes (df, analysis) and returns updated analysis dict.
        """
        self.middlewares.append(middleware)

    def run(self) -> AnalysisDict:
        """
        Run all registered middlewares in sequence, updating the analysis dict.
        Returns:
            AnalysisDict: Dictionary containing results from all middlewares.
        """
        analysis: AnalysisDict = {}
        for middleware in self.middlewares:
            try:
                result = middleware(self.time_frame, self.df, self.last_pivot, analysis, self.useLogScale)
                # Each middleware should return {middleware_name: {lines: [], pivots: [], vp: []}}
                for key, value in result.items():
                    analysis[key] = value
            except Exception as e:
                # Log error but continue with next middleware
                print(f"Error occurred while processing middleware {middleware.__name__}: {e}")
                pass

        return analysis
