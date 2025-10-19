"""
Clean Trading Environment for Reinforcement Learning
Uses Broker class for all trading operations
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Dict

from data_processing.feature_normalizer import FeatureNormalizer
from .broker import TradingBroker


class TradingEnvironment(gym.Env):
    """Clean trading environment using Broker for trading operations"""

    def __init__(self,
                 df: pd.DataFrame,
                 normalizer: FeatureNormalizer,
                 feature_config: Dict[str, str],
                 initial_balance: float = 1000000.0,
                 window_size: int = 672,
                 buy_threshold: float = 0.1,
                 sell_threshold: float = -0.1):

        super().__init__()

        # Core parameters
        self.df = df.copy().reset_index(drop=True)
        self.normalizer = normalizer
        self.feature_config = feature_config
        self.feature_columns = list(feature_config.keys())
        self.window_size = window_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Validate DataFrame
        missing_features = set(self.feature_columns) - set(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Initialize broker
        self.broker = TradingBroker(initial_balance)

        # Environment state
        self.current_step = window_size

        # Prepare data
        self._prepare_data()

        # Define spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )

        obs_dim = len(self.feature_columns) + 10  # Features + account info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, obs_dim),
            dtype=np.float32
        )

    def _prepare_data(self):
        """Prepare normalized data"""
        print("⚡ Pre-normalizing feature data...")
        normalized_features = self.normalizer.transform(
            self.df[self.feature_columns],
            close_prices=self.df['close']
        )
        self.normalized_features = normalized_features[self.feature_columns].values
        print(f"✅ Pre-normalized {len(self.feature_columns)} features")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and broker"""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.broker.reset()

        return self._get_obs(), {}

    def step(self, action):
        """Execute one step using broker"""
        # Decode action
        raw_signal, raw_position_size = action
        signal, position_size = self._decode_action(raw_signal, raw_position_size)

        # Get current price
        current_price = self.df['close'][self.current_step]

        # Execute trade through broker
        step_pnl, trade_occurred, is_bankrupt = self.broker.step(
            signal, position_size, current_price, self.current_step
        )

        # Check termination conditions - STOP training on bankruptcy
        terminated = is_bankrupt
        truncated = self.current_step >= len(self.df) - 1

        # Calculate reward - let positive PnL guide learning
        if terminated:
            # End episode on bankruptcy with neutral reward
            reward = 0.0
        else:
            # Normal reward calculation based on PnL
            reward = self._calculate_reward(step_pnl, current_price)

        # Move to next step
        self.current_step += 1

        # Create info
        info = self._create_info(signal, position_size, current_price,
                                 step_pnl,  trade_occurred,
                                 is_bankrupt)

        return self._get_obs(), reward, terminated, truncated, info

    def _decode_action(self, raw_signal, position_size):
        """Convert raw action to trading signal"""
        if raw_signal >= self.buy_threshold:
            return 1, position_size  # BUY
        elif raw_signal <= self.sell_threshold:
            return -1, position_size  # SELL
        else:
            return 0, 0.0  # HOLD

    def _calculate_reward(self, step_pnl, current_price):
        """Simple percentage-based reward - proven to work"""

        total_portfolio = self.broker.balance + self.broker.capital_used
        initial_balance = self.broker.initial_balance

        # Simple percentage returns (already normalized to 0-1 range)
        portfolio_pct = (total_portfolio - initial_balance) / initial_balance
        step_pct = step_pnl / initial_balance

        # Balanced: care about both step and cumulative performance
        reward = step_pct + 0.1 * portfolio_pct  # Step has 10x more weight than cumulative

        # Clip to safe range
        return np.clip(reward, -0.5, 0.5)

    def _get_obs(self):
        """Get observation window - fully vectorized for LSTM"""
        end_step = self.current_step
        start_step = end_step - self.window_size + 1
        steps = np.arange(max(0, start_step), end_step + 1)
        actual_window_size = len(steps)

        # Clamp feature access
        batch_features = self.normalized_features[np.clip(steps, 0, len(self.normalized_features) - 1)]
        current_prices = self.df['close'].iloc[np.clip(steps, 0, len(self.df) - 1)].values
        initial_balance = self.broker.initial_balance

        # Gather broker states
        broker_states = []
        for step, price in zip(steps, current_prices):
            history_index = step - self.window_size
            if 0 <= history_index < len(self.broker.step_history):
                state = self.broker.step_history[history_index]
            else:
                state = self.broker.create_step_state(step, price, 0, 0.0, 0.0, False)
            broker_states.append(state)

        # Vectorized normalization
        position_size = np.array([s['position_size'] for s in broker_states], dtype=np.float32)
        signal = np.array([s['signal'] for s in broker_states], dtype=np.float32)
        position_sign = np.sign(np.array([s['position_shares'] for s in broker_states]))
        position_shares = np.array([s['position_shares'] for s in broker_states], dtype=np.float32)
        position_shares_scaled = position_shares / self.broker.quantity_precision

        step_pnl = np.array([s['step_pnl'] for s in broker_states], dtype=np.float32) / initial_balance
        portfolio_ratio = np.array([s['portfolio_value'] for s in broker_states], dtype=np.float32) / initial_balance
        capital_available = np.array([s['balance'] for s in broker_states], dtype=np.float32) / initial_balance
        capital_used = np.array([s['capital_used'] for s in broker_states], dtype=np.float32) / initial_balance
        total_trades = np.tanh(np.array([s['total_trades'] for s in broker_states]) / 10000.0)

        # Assemble account info
        account_info = np.stack([
            signal,
            position_size,
            portfolio_ratio,
            capital_available,
            position_sign,
            position_shares_scaled,
            step_pnl,
            total_trades,
            capital_used,
            np.ones(actual_window_size, dtype=np.float32)  # bias term
        ], axis=1)

        # Concatenate features + broker info
        obs_array = np.concatenate([batch_features, account_info], axis=1)

        # Pad at beginning if window < window_size
        if obs_array.shape[0] < self.window_size:
            pad_rows = self.window_size - obs_array.shape[0]
            obs_array = np.vstack([np.zeros((pad_rows, obs_array.shape[1]), dtype=np.float32), obs_array])

        return obs_array.astype(np.float32)

    def _create_info(self, signal, position_size, price, step_pnl, trade_occurred,
                     is_bankrupt):
        """Create info dictionary with action validation feedback"""
        broker_state = self.broker.get_state()

        # Add action validation info
        return {
            'signal': signal,
            'position_size': position_size,
            'price': price,
            'step_pnl': step_pnl,
            'trade_occurred': trade_occurred,
            'is_bankrupt': is_bankrupt,
            **broker_state  # Include all broker state
        }
