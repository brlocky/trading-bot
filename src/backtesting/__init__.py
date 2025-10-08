"""
Backtesting module for trading bot using VectorBT.

This module provides backtesting functionality using VectorBT
for vectorized, high-performance strategy testing.

Uses AutonomousTrader as the single source of truth for feature calculation.
"""

from .vectorbt_engine import VectorBTBacktester

__all__ = ['VectorBTBacktester']
