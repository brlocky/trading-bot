"""Quick test of ExpertTradeReplay system"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional


class ExpertTradeReplay:
    """Copied minimal version for testing"""

    def __init__(self, filepath: str = 'expert_trades.pkl', max_trades: int = 1000):
        self.filepath = filepath
        self.max_trades = max_trades
        self.expert_trades: List[Dict] = []
        self.load()

    def load(self) -> bool:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    self.expert_trades = pickle.load(f)
                print(f"OK Loaded {len(self.expert_trades)} expert trades")
                return True
            except Exception as e:
                print(f"WARN Failed to load: {e}")
                self.expert_trades = []
                return False
        else:
            self.expert_trades = []
            return False

    def save(self) -> bool:
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.expert_trades, f)
            print(f"OK Saved {len(self.expert_trades)} expert trades")
            return True
        except Exception as e:
            print(f"WARN Failed to save: {e}")
            return False

    def record_trade(self, entry_step, exit_step, trade_object, actions, pnl_percent, hold_duration, timestamp=None, min_pnl=2.0, max_duration=50):
        if pnl_percent > min_pnl and hold_duration < max_duration:
            expert_trade = {
                'entry_step': entry_step,
                'exit_step': exit_step,
                'trade_object': trade_object.copy() if hasattr(trade_object, 'copy') else dict(trade_object),
                'actions': actions.copy() if actions else [],
                'pnl_percent': pnl_percent,
                'hold_duration': hold_duration,
                'timestamp': timestamp
            }
            self.expert_trades.append(expert_trade)
            if len(self.expert_trades) > self.max_trades:
                self.expert_trades = self.expert_trades[-self.max_trades:]
            return True
        return False

    def get_random_trade(self) -> Optional[Dict]:
        if not self.expert_trades:
            return None
        idx = np.random.randint(0, len(self.expert_trades))
        return self.expert_trades[idx]

    def should_inject(self, injection_rate: float = 0.2) -> bool:
        return len(self.expert_trades) > 0 and np.random.rand() < injection_rate

    def get_statistics(self) -> Dict:
        if not self.expert_trades:
            return {'total_trades': 0, 'avg_pnl': 0, 'avg_duration': 0}
        pnls = [t['pnl_percent'] for t in self.expert_trades]
        durations = [t['hold_duration'] for t in self.expert_trades]
        return {
            'total_trades': len(self.expert_trades),
            'avg_pnl': np.mean(pnls),
            'avg_duration': np.mean(durations),
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls)
        }

    def __len__(self):
        return len(self.expert_trades)


# Test basic functionality
replay = ExpertTradeReplay(filepath='test_expert_trades.pkl', max_trades=10)

print("Initial state:")
print(f"  Trades loaded: {len(replay)}")
print(f"  Stats: {replay.get_statistics()}")
print()

# Add some test trades
print("Adding test trades...")
for i in range(5):
    success = replay.record_trade(
        entry_step=1000 + i*100,
        exit_step=1025 + i*100,
        trade_object={
            'status': 'CLOSED',
            'entry_price': 45000 + i*100,
            'exit_price': 46000 + i*100,
            'pnl': 100 + i*10,
            'pnl_percent': 2.5 + i*0.5
        },
        actions=[
            {'step': 1000 + i*100, 'action': [1, 5, 3], 'close_price': 45000},
            {'step': 1025 + i*100, 'action': [3, 0, 0], 'close_price': 46000}
        ],
        pnl_percent=2.5 + i*0.5,
        hold_duration=25,
        timestamp=f"2024-03-{15+i} 14:00:00"
    )
    print(f"  Trade {i+1}: {'OK' if success else 'SKIP'} (PnL: {2.5 + i*0.5:.1f}%)")

print()
print("After adding trades:")
print(f"  Total trades: {len(replay)}")
print(f"  Stats: {replay.get_statistics()}")
print()

# Test injection logic
print("Testing injection logic:")
for i in range(10):
    if replay.should_inject(0.2):
        trade = replay.get_random_trade()
        if trade:
            print(f"  Trial {i+1}: INJECT - Entry step: {trade['entry_step']}, PnL: {trade['pnl_percent']:.1f}%")
        else:
            print(f"  Trial {i+1}: INJECT but no trade available")
    else:
        print(f"  Trial {i+1}: Normal random start")

print()

# Test save/load
print("Testing save/load:")
replay.save()
replay2 = ExpertTradeReplay(filepath='test_expert_trades.pkl')
print(f"  Loaded {len(replay2)} trades")
print(f"  Stats match: {replay.get_statistics() == replay2.get_statistics()}")

print()
print("OK All tests passed!")

# Cleanup
if os.path.exists('test_expert_trades.pkl'):
    os.remove('test_expert_trades.pkl')
    print("OK Cleaned up test file")
