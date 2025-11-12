"""
Trade Logger - Save all trades for post-training analysis
"""
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TradeLogger:
    """
    Logs all trades during training for later analysis and labeling.
    Saves to PKL file on environment reset.
    """

    def __init__(self, log_dir: str = "logs/trades", max_trades: int = 10000):
        """
        Args:
            log_dir: Directory to save trade logs
            max_trades: Maximum trades to keep in memory before auto-saving
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_trades = max_trades

        # Current session trades
        self.trades: List[Dict] = []

        # Total trades logged this session
        self.total_logged = 0

    def log_trade(self, trade_data: Dict):
        """
        Log a single trade with full context.

        Expected trade_data keys:
            - timestamp: int (candle index)
            - action: str ('LONG', 'SHORT', 'CLOSE')
            - entry_price: float
            - exit_price: float (if closed)
            - size: float
            - pnl: float (if closed)
            - pnl_pct: float (if closed)
            - duration: int (candles held)

            # Market context at entry
            - close: float
            - vah: float
            - poc: float
            - val: float
            - visible_high: float
            - visible_low: float

            # Distance features (normalized -1 to 1)
            - dist_to_vah: float
            - dist_to_poc: float
            - dist_to_val: float

            # Binary features
            - close_in_va: bool
            - close_above_va: bool
            - close_below_va: bool
            - close_above_poc: bool

            # Technical context
            - rsi: float
            - volume: float
            - atr: float

            # Outcome (to be labeled later)
            - label: Optional[str] = None  # 'good_entry', 'bad_entry', 'neutral'
            - label_reason: Optional[str] = None  # Why it was labeled
        """
        self.trades.append(trade_data)

        # Auto-save if buffer full
        if len(self.trades) >= self.max_trades:
            self.save_trades()

    def save_trades(self, filename: Optional[str] = None):
        """
        Save accumulated trades to PKL file.

        Args:
            filename: Optional custom filename. If None, uses timestamp.
        """
        if len(self.trades) == 0:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{timestamp}.pkl"

        filepath = self.log_dir / filename

        # Load existing trades if file exists (append mode)
        existing_trades = []
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    existing_trades = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing trades: {e}")

        # Merge and save
        all_trades = existing_trades + self.trades

        with open(filepath, 'wb') as f:
            pickle.dump(all_trades, f)

        self.total_logged += len(self.trades)
        print(f"💾 Saved {len(self.trades)} trades to {filepath} (Total: {self.total_logged})")

        # Clear buffer
        self.trades = []

    def load_trades(self, filename: str) -> List[Dict]:
        """
        Load trades from PKL file.

        Args:
            filename: Name of PKL file in log_dir

        Returns:
            List of trade dictionaries
        """
        filepath = self.log_dir / filename

        if not filepath.exists():
            print(f"Warning: {filepath} does not exist")
            return []

        try:
            with open(filepath, 'rb') as f:
                trades = pickle.load(f)
            print(f"✓ Loaded {len(trades)} trades from {filepath}")
            return trades
        except Exception as e:
            print(f"Error loading trades: {e}")
            return []

    def get_all_trade_files(self) -> List[str]:
        """Get list of all trade log files."""
        return sorted([f.name for f in self.log_dir.glob("trades_*.pkl")])

    def get_stats(self) -> Dict:
        """Get statistics about current trade buffer."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'longs': 0,
                'shorts': 0,
                'closed': 0
            }

        longs = sum(1 for t in self.trades if t.get('action') == 'LONG')
        shorts = sum(1 for t in self.trades if t.get('action') == 'SHORT')
        closed = sum(1 for t in self.trades if t.get('exit_price') is not None)

        return {
            'total_trades': len(self.trades),
            'longs': longs,
            'shorts': shorts,
            'closed': closed,
            'total_logged_session': self.total_logged
        }
