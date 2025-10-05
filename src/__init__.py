"""
Trading Bot - Main Package
Contains all core trading bot modules and utilities.
"""

__version__ = "1.0.0"

# Optional: Export commonly used classes for easier imports
# This allows: from src import SimpleModelTrainer
# Instead of: from src.training.model_trainer import SimpleModelTrainer

from .training.model_trainer import SimpleModelTrainer
from .training.autonomous_trainer import AutonomousTraderTrainer
from .prediction.predictor import SimpleModelPredictor
from .prediction.reporter import SimpleModelReporter
from .trading.autonomous_trader import AutonomousTrader
from .memory.trade_memory import TradeMemoryManager
from .core.trading_types import TradingAction, TradingSignal, LevelInfo, TradeRecord

__all__ = [
    'SimpleModelTrainer',
    'AutonomousTraderTrainer',
    'SimpleModelPredictor',
    'SimpleModelReporter',
    'AutonomousTrader',
    'TradeMemoryManager',
    'TradingAction',
    'TradingSignal',
    'LevelInfo',
    'TradeRecord',
]
