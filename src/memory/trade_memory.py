"""
Trade Memory Manager - Tracks and analyzes past trades
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List

from core.trading_types import TradeRecord


class TradeMemoryManager:
    """
    🧠 TRADE MEMORY MANAGER - Advanced Trade Tracking & Learning
    ==========================================================

    Tracks and analyzes past trades to improve future predictions:
    - Win/loss statistics by timeframe and conditions
    - Support/resistance bounce success rates  
    - Recent performance trends
    - Pattern recognition for similar market conditions
    """

    def __init__(self, max_memory: int = 1000):
        self.max_memory = max_memory
        self.trades: List[TradeRecord] = []
        self.memory_file = "trade_memory.json"
        self.load_memory()

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a new trade to memory"""
        self.trades.append(trade)

        # Keep only recent trades
        if len(self.trades) > self.max_memory:
            self.trades = self.trades[-self.max_memory:]

        self.save_memory()

    def get_recent_performance(self, lookback_hours: int = 168) -> Dict[str, float]:
        """Get performance stats for recent trades (default: last 7 days)"""
        if not self.trades:
            return {'win_rate': 0.5, 'avg_pnl': 0.0, 'total_trades': 0}

        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=lookback_hours)
        recent_trades = [t for t in self.trades if t.timestamp >= cutoff_time and t.pnl_pct is not None]

        if not recent_trades:
            return {'win_rate': 0.5, 'avg_pnl': 0.0, 'total_trades': 0}

        winning_trades = [t for t in recent_trades if t.pnl_pct > 0]
        win_rate = len(winning_trades) / len(recent_trades)
        avg_pnl = np.mean([t.pnl_pct for t in recent_trades])

        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_trades': len(recent_trades)
        }

    def get_bounce_performance(self) -> Dict[str, float]:
        """Get performance stats specifically for bounce trades"""
        bounce_trades = [t for t in self.trades if t.was_bounce and t.pnl_pct is not None]

        if not bounce_trades:
            return {'bounce_win_rate': 0.6, 'bounce_avg_pnl': 0.5, 'bounce_trades': 0}

        winning_bounces = [t for t in bounce_trades if t.pnl_pct > 0]
        bounce_win_rate = len(winning_bounces) / len(bounce_trades)
        bounce_avg_pnl = np.mean([t.pnl_pct for t in bounce_trades])

        return {
            'bounce_win_rate': bounce_win_rate,
            'bounce_avg_pnl': bounce_avg_pnl,
            'bounce_trades': len(bounce_trades)
        }

    def get_consecutive_performance(self) -> int:
        """Get current streak of wins/losses"""
        if not self.trades:
            return 0

        # Get recent completed trades
        completed_trades = [t for t in self.trades if t.pnl_pct is not None]
        if not completed_trades:
            return 0

        # Count consecutive wins from the most recent trade
        consecutive = 0
        for trade in reversed(completed_trades):
            if trade.pnl_pct > 0:  # Win
                consecutive += 1
            else:  # Loss - break the streak
                break

        return consecutive

    def save_memory(self) -> None:
        """Save trade memory to disk"""
        try:
            # Convert trades to serializable format
            trade_data = []
            for trade in self.trades:
                trade_dict = {
                    'timestamp': trade.timestamp.isoformat() if trade.timestamp else None,
                    'signal_type': trade.signal_type,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl_pct': trade.pnl_pct,
                    'duration_hours': trade.duration_hours,
                    'was_bounce': trade.was_bounce,
                    'bounce_level_price': trade.bounce_level_price,
                    'bounce_level_type': trade.bounce_level_type,
                    'confidence': trade.confidence
                }
                trade_data.append(trade_dict)

            with open(self.memory_file, 'w') as f:
                json.dump(trade_data, f, indent=2)

        except Exception as e:
            print(f"⚠️ Warning: Could not save trade memory: {e}")

    def load_memory(self) -> None:
        """Load trade memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    trade_data = json.load(f)

                # Convert back to TradeRecord objects
                self.trades = []
                for trade_dict in trade_data:
                    timestamp = pd.Timestamp(trade_dict['timestamp']) if trade_dict['timestamp'] else None
                    trade = TradeRecord(
                        timestamp=timestamp,
                        signal_type=trade_dict['signal_type'],
                        entry_price=trade_dict['entry_price'],
                        exit_price=trade_dict.get('exit_price'),
                        pnl_pct=trade_dict.get('pnl_pct'),
                        duration_hours=trade_dict.get('duration_hours'),
                        was_bounce=trade_dict.get('was_bounce', False),
                        bounce_level_price=trade_dict.get('bounce_level_price'),
                        bounce_level_type=trade_dict.get('bounce_level_type'),
                        confidence=trade_dict.get('confidence')
                    )
                    self.trades.append(trade)

                print(f"📚 Loaded {len(self.trades)} trades from memory")

        except Exception as e:
            print(f"⚠️ Warning: Could not load trade memory: {e}")
            self.trades = []
