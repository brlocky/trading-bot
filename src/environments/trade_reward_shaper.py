"""
Trade-Based Reward Shaping

Uses labeled trades to reward VP-aware entries:
- Reward entries similar to labeled "good" entries
- Penalize entries similar to labeled "bad" entries
"""
import pickle
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class TradeRewardShaper:
    """
    Shapes rewards based on labeled trade examples.

    Rewards entries that match "good" trade patterns (e.g., buying near VAL)
    Penalizes entries that match "bad" trade patterns (e.g., buying at VAH)
    """

    def __init__(self, labeled_trades_path: Optional[str] = None):
        """
        Args:
            labeled_trades_path: Path to PKL file with labeled trades
        """
        self.good_entries = []
        self.bad_entries = []

        if labeled_trades_path:
            self.load_labeled_trades(labeled_trades_path)

    def load_labeled_trades(self, filepath: str):
        """Load and categorize labeled trades."""
        with open(filepath, 'rb') as f:
            trades = pickle.load(f)

        # Separate by label
        self.good_entries = [t for t in trades if t.get('label') == 'good_entry']
        self.bad_entries = [t for t in trades if t.get('label') == 'bad_entry']

        print(f"✓ Loaded {len(self.good_entries)} good entries, {len(self.bad_entries)} bad entries")

        # Calculate average VP distances for good/bad entries
        self._calculate_patterns()

    def _calculate_patterns(self):
        """Calculate typical VP distance patterns for good/bad entries."""
        if len(self.good_entries) > 0:
            # Good LONG entries (should be near VAL/POC support)
            good_longs = [t for t in self.good_entries if t['action'] == 'LONG']
            if good_longs:
                self.good_long_pattern = {
                    'avg_dist_to_val': np.mean([t['dist_to_val'] for t in good_longs]),
                    'avg_dist_to_poc': np.mean([t['dist_to_poc'] for t in good_longs]),
                    'avg_dist_to_vah': np.mean([t['dist_to_vah'] for t in good_longs]),
                    'in_va_rate': np.mean([t['close_in_va'] for t in good_longs]),
                    'below_poc_rate': np.mean([not t['close_above_poc'] for t in good_longs]),
                }
            else:
                self.good_long_pattern = None

            # Good SHORT entries (should be near VAH/POC resistance)
            good_shorts = [t for t in self.good_entries if t['action'] == 'SHORT']
            if good_shorts:
                self.good_short_pattern = {
                    'avg_dist_to_vah': np.mean([t['dist_to_vah'] for t in good_shorts]),
                    'avg_dist_to_poc': np.mean([t['dist_to_poc'] for t in good_shorts]),
                    'avg_dist_to_val': np.mean([t['dist_to_val'] for t in good_shorts]),
                    'in_va_rate': np.mean([t['close_in_va'] for t in good_shorts]),
                    'above_poc_rate': np.mean([t['close_above_poc'] for t in good_shorts]),
                }
            else:
                self.good_short_pattern = None

        if len(self.bad_entries) > 0:
            # Bad LONG entries (buying at resistance)
            bad_longs = [t for t in self.bad_entries if t['action'] == 'LONG']
            if bad_longs:
                self.bad_long_pattern = {
                    'avg_dist_to_vah': np.mean([t['dist_to_vah'] for t in bad_longs]),
                    'above_poc_rate': np.mean([t['close_above_poc'] for t in bad_longs]),
                }
            else:
                self.bad_long_pattern = None

            # Bad SHORT entries (selling at support)
            bad_shorts = [t for t in self.bad_entries if t['action'] == 'SHORT']
            if bad_shorts:
                self.bad_short_pattern = {
                    'avg_dist_to_val': np.mean([t['dist_to_val'] for t in bad_shorts]),
                    'below_poc_rate': np.mean([not t['close_above_poc'] for t in bad_shorts]),
                }
            else:
                self.bad_short_pattern = None

    def calculate_entry_quality_reward(self, action: str, vp_features: Dict) -> float:
        """
        Calculate reward bonus/penalty based on VP context similarity to labeled trades.

        Args:
            action: 'LONG' or 'SHORT'
            vp_features: Dict with keys:
                - dist_to_vah, dist_to_poc, dist_to_val (floats, -1 to 1)
                - close_in_va, close_above_poc (bools)

        Returns:
            Reward: +0.5 to -0.5 based on entry quality
        """
        if len(self.good_entries) == 0 and len(self.bad_entries) == 0:
            return 0.0  # No labeled data, no shaping

        reward = 0.0

        if action == 'LONG':
            # Check similarity to good LONG patterns
            if self.good_long_pattern:
                # Good longs are typically:
                # - Near VAL (dist_to_val close to 0, negative = below)
                # - Below or at POC (not above POC)
                # - In Value Area

                # Reward being near VAL
                if vp_features['dist_to_val'] < 0.1 and vp_features['dist_to_val'] > -0.2:
                    reward += 0.3  # Near VAL support

                # Reward being below POC (looking to buy support)
                if not vp_features['close_above_poc']:
                    reward += 0.2

            # Check similarity to bad LONG patterns
            if self.bad_long_pattern:
                # Bad longs are typically:
                # - Near VAH (buying at resistance)
                # - Way above POC

                # Penalize buying near VAH
                if vp_features['dist_to_vah'] < 0.1 and vp_features['dist_to_vah'] > -0.1:
                    reward -= 0.3  # At VAH resistance

                # Penalize buying way above POC
                if vp_features['close_above_poc'] and vp_features['dist_to_poc'] > 0.3:
                    reward -= 0.2

        elif action == 'SHORT':
            # Check similarity to good SHORT patterns
            if self.good_short_pattern:
                # Good shorts are typically:
                # - Near VAH (dist_to_vah close to 0)
                # - Above POC

                # Reward being near VAH
                if vp_features['dist_to_vah'] < 0.1 and vp_features['dist_to_vah'] > -0.1:
                    reward += 0.3  # Near VAH resistance

                # Reward being above POC
                if vp_features['close_above_poc']:
                    reward += 0.2

            # Check similarity to bad SHORT patterns
            if self.bad_short_pattern:
                # Bad shorts are typically:
                # - Near VAL (selling at support)
                # - Below POC

                # Penalize shorting near VAL
                if vp_features['dist_to_val'] < 0.1 and vp_features['dist_to_val'] > -0.1:
                    reward -= 0.3  # At VAL support

                # Penalize shorting below POC
                if not vp_features['close_above_poc']:
                    reward -= 0.2

        return np.clip(reward, -0.5, 0.5)

    def get_stats(self) -> Dict:
        """Get statistics about loaded patterns."""
        stats = {
            'total_good_entries': len(self.good_entries),
            'total_bad_entries': len(self.bad_entries),
        }

        if hasattr(self, 'good_long_pattern') and self.good_long_pattern:
            stats['good_long_pattern'] = self.good_long_pattern

        if hasattr(self, 'good_short_pattern') and self.good_short_pattern:
            stats['good_short_pattern'] = self.good_short_pattern

        if hasattr(self, 'bad_long_pattern') and self.bad_long_pattern:
            stats['bad_long_pattern'] = self.bad_long_pattern

        if hasattr(self, 'bad_short_pattern') and self.bad_short_pattern:
            stats['bad_short_pattern'] = self.bad_short_pattern

        return stats


# Example integration with SimpleTradingEnv
"""
To use in your environment's calculate_reward method:

# In __init__:
self.trade_reward_shaper = TradeRewardShaper('logs/trades/trades_labeled.pkl')

# In calculate_reward, after opening a position:
if current_state['position_type'] != 'FLAT' and previous_state['position_type'] == 'FLAT':
    # Position just opened
    action_name = 'LONG' if current_state['position_type'] == 'LONG' else 'SHORT'
    
    # Get current VP features
    vp_levels = get_vp_levels_features_visible(self.data, self.current_step, self.lookback_window, self.n_bins)
    last_vp = vp_levels[-1]  # Last timestep
    
    vp_context = {
        'dist_to_vah': float(last_vp[4]),
        'dist_to_poc': float(last_vp[8]),
        'dist_to_val': float(last_vp[12]),
        'close_in_va': bool(last_vp[20] > 0.5),
        'close_above_poc': bool(last_vp[24] > 0.5),
    }
    
    # Get entry quality reward
    entry_quality = self.trade_reward_shaper.calculate_entry_quality_reward(action_name, vp_context)
    
    # Add to total reward (scale appropriately)
    reward += entry_quality * 0.1  # 10% weight for entry quality
"""
