"""
Core Trading Types - Data structures for trading system
"""

import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from src.ta.technical_analysis import ChartInterval


class TradingAction(Enum):
    """Trading actions the model can take"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradingSignal:
    """Trading signal with reasoning"""
    action: TradingAction
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    risk_reward_ratio: Optional[float] = None


@dataclass
class LevelInfo:
    """Information about a support/resistance level"""
    price: float
    strength: float  # How many times it was tested
    distance: float  # Distance from current price (%)
    level_type: str  # 'support', 'resistance', 'poc', 'vah', 'val', etc.
    timeframe: ChartInterval
    last_test_time: Optional[pd.Timestamp] = None


@dataclass
class TradeRecord:
    """Single trade record for memory system"""
    timestamp: pd.Timestamp
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    entry_price: float
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_hours: Optional[int] = None
    was_bounce: bool = False
    bounce_level_price: Optional[float] = None
    bounce_level_type: Optional[str] = None
    confidence: Optional[float] = None
