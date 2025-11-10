"""
Pattern Memory for PPO Trading Bot
Stores episodes and transitions for analysis and curriculum learning
Compatible with PPO (on-policy) by storing complete episodes
"""
import numpy as np
import pickle
from collections import deque
from pathlib import Path
import pandas as pd


class TradingPatternMemory:
    """
    Stores trading patterns for post-training analysis and curriculum learning.
    Compatible with PPO (on-policy) by storing complete episodes.
    """

    def __init__(self, capacity=1000, save_dir='data/pattern_memory'):
        self.capacity = capacity
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Store complete episodes
        self.episodes = deque(maxlen=capacity)

        # Statistics tracking
        self.stats = {
            'total_episodes': 0,
            'winning_episodes': 0,
            'losing_episodes': 0,
            'avg_return': 0.0,
            'best_return': -np.inf,
            'worst_return': np.inf
        }

    def add_episode(self, episode_data):
        """
        Store a complete episode with all transitions

        Args:
            episode_data: dict with keys:
                - transitions: list of (state, action, reward, next_state, done, info)
                - total_return: sum of rewards
                - total_trades: number of trades taken
                - win_rate: percentage of winning trades
                - final_balance: ending balance
                - episode_length: number of steps
                - market_conditions: dict with volatility, trend, etc.
        """
        self.episodes.append(episode_data)

        # Update statistics
        self.stats['total_episodes'] += 1
        total_return = episode_data['total_return']

        if total_return > 0:
            self.stats['winning_episodes'] += 1
        else:
            self.stats['losing_episodes'] += 1

        self.stats['best_return'] = max(self.stats['best_return'], total_return)
        self.stats['worst_return'] = min(self.stats['worst_return'], total_return)

        # Running average
        n = self.stats['total_episodes']
        self.stats['avg_return'] = (
            (self.stats['avg_return'] * (n - 1) + total_return) / n
        )

    def get_top_episodes(self, n=10, criterion='return'):
        """
        Get top N episodes by criterion

        Args:
            n: number of episodes to return
            criterion: 'return', 'win_rate', 'balance', 'trades'
        """
        if len(self.episodes) == 0:
            return []

        # Map criterion names to dict keys
        key_map = {
            'return': 'total_return',
            'win_rate': 'win_rate',
            'balance': 'final_balance',
            'trades': 'total_trades'
        }

        sort_key = key_map.get(criterion, 'total_return')

        sorted_episodes = sorted(
            self.episodes,
            key=lambda x: x.get(sort_key, 0),
            reverse=True
        )
        return sorted_episodes[:n]

    def get_pattern_distribution(self):
        """
        Analyze what patterns led to success

        Returns:
            dict with pattern statistics
        """
        if len(self.episodes) == 0:
            return {}

        winning_episodes = [ep for ep in self.episodes if ep['total_return'] > 0]
        losing_episodes = [ep for ep in self.episodes if ep['total_return'] <= 0]

        def analyze_episodes(episodes):
            if not episodes:
                return {}

            return {
                'avg_trades': np.mean([ep['total_trades'] for ep in episodes]),
                'avg_length': np.mean([ep['episode_length'] for ep in episodes]),
                'avg_win_rate': np.mean([ep['win_rate'] for ep in episodes]),
                'avg_return': np.mean([ep['total_return'] for ep in episodes]),
                'avg_final_balance': np.mean([ep['final_balance'] for ep in episodes]),
            }

        return {
            'winning_patterns': analyze_episodes(winning_episodes),
            'losing_patterns': analyze_episodes(losing_episodes),
            'total_episodes': len(self.episodes),
            'win_rate': len(winning_episodes) / len(self.episodes) if self.episodes else 0
        }

    def save(self, filename='pattern_memory.pkl'):
        """Save memory to disk"""
        filepath = self.save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'episodes': list(self.episodes),
                'stats': self.stats
            }, f)
        print(f"✓ Saved pattern memory: {filepath}")

    def load(self, filename='pattern_memory.pkl'):
        """Load memory from disk"""
        filepath = self.save_dir / filename
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.episodes = deque(data['episodes'], maxlen=self.capacity)
                self.stats = data['stats']
            print(f"✓ Loaded pattern memory: {filepath} ({len(self.episodes)} episodes)")
            return True
        except FileNotFoundError:
            print(f"⚠ No saved memory found at {filepath}")
            return False

    def export_to_dataframe(self):
        """Export episode summaries to pandas DataFrame for analysis"""
        if len(self.episodes) == 0:
            return pd.DataFrame()

        records = []
        for i, ep in enumerate(self.episodes):
            records.append({
                'episode_id': i,
                'total_return': ep['total_return'],
                'final_balance': ep['final_balance'],
                'total_trades': ep['total_trades'],
                'win_rate': ep['win_rate'],
                'episode_length': ep['episode_length'],
                'avg_reward': ep['total_return'] / ep['episode_length'] if ep['episode_length'] > 0 else 0
            })

        return pd.DataFrame(records)

    def get_statistics(self):
        """Get summary statistics"""
        return {
            'total_episodes': len(self.episodes),
            'winning_episodes': self.stats['winning_episodes'],
            'losing_episodes': self.stats['losing_episodes'],
            'win_rate': self.stats['winning_episodes'] / max(1, self.stats['total_episodes']),
            'avg_return': self.stats['avg_return'],
            'best_return': self.stats['best_return'] if self.stats['best_return'] != -np.inf else 0.0,
            'worst_return': self.stats['worst_return'] if self.stats['worst_return'] != np.inf else 0.0,
        }
