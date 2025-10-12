"""
Core Trading Types - Data structures for trading system
"""

from typing_extensions import Literal
import pandas as pd
from dataclasses import dataclass
from typing import Optional

ChartInterval = Literal[
    '1m', '2m', '3m', '5m', '10m', '15m', '30m',
    '1h', '2h', '4h', 'D', 'W', 'M'
]


@dataclass
class LevelInfo:
    """Information about a support/resistance level"""
    price: float
    strength: float  # How many times it was tested
    distance: float  # Distance from current price (%)
    level_type: str  # 'support', 'resistance', 'poc', 'vah', 'val', etc.
    timeframe: ChartInterval
    last_test_time: Optional[pd.Timestamp] = None
