"""
Clean Trading Environment for Reinforcement Learning
Uses Broker class for all trading operations
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional

from data_processing.add_features import get_indicator_features, get_price_features, get_vp_levels
from environments.trade_reward import TradeReward
from .broker import TradingBroker


class TradingEnvironment(gym.Env):
    """Clean trading environment using Broker for trading operations"""

    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 1000000.0,
                 window_size: int = 672,
                 buy_threshold: float = 0.1,
                 sell_threshold: float = -0.1):

        super().__init__()

        # Core parameters
        self.window_size = window_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Initialize broker
        self.broker = TradingBroker(initial_balance)
        self.trade_count = 0
        self.last_broker_state = None

        # Environment state
        self.current_step = window_size

        self.step_history = []

        # Precompute features
        self.df = df.copy().reset_index(drop=True)
        self.indicator_features = get_indicator_features(self.df)

        removed_rows = len(self.df) - len(self.indicator_features)
        if removed_rows > 0:
            self.df = self.df.iloc[removed_rows:].reset_index(drop=True)
            print(f"TradingEnvironment: Aligned df and indicator_features by removing first {removed_rows} rows.")

        self.price_features = get_price_features(self.df)

        self.daily_vp = {}
        self.prev_day_vp = {}
        self.weekly_vp = {}

        # Define spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'price': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.price_features.shape[1]),
                dtype=np.float32
            ),
            'features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.indicator_features.shape[1]),
                dtype=np.float32
            ),
            'volume_profile_week': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, 3),  # Matches account_info stack (8 elements)
                dtype=np.float32
            ),
            'volume_profile_prev_day': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, 3),  # Matches account_info stack (8 elements)
                dtype=np.float32
            ),
            'volume_profile_daily': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, 3),  # Matches account_info stack (8 elements)
                dtype=np.float32
            ),
            'account': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, 6),  # Matches account_info stack (6 elements)
                dtype=np.float32
            )
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and broker"""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.step_history = []
        self.trade_count = 0
        self.last_broker_state = None
        self.broker.reset()
        self.daily_vp = {}
        self.prev_day_vp = {}
        self.weekly_vp = {}

        return self._get_obs(), {}

    def step(self, action):
        """Execute one step using broker"""
        # Decode action
        raw_signal, raw_position_size = action
        signal, position_size = self._decode_action(raw_signal, raw_position_size)

        # Update last broker state for reward calculation
        self.last_broker_state = self.broker.get_state() if self.broker.step_history else None

        # Get current price
        current_price = self.df['close'][self.current_step]

        # Execute trade through broker
        _, _, is_bankrupt = self.broker.step(
            signal, position_size, current_price, self.current_step
        )

        # Check termination conditions - STOP training on bankruptcy
        terminated = is_bankrupt

        # Calculate reward - let positive PnL guide learning
        if terminated:
            # End episode on bankruptcy with neutral reward
            reward = 0.0
        else:
            # Normal reward calculation based on PnL
            reward = self._calculate_reward(current_price)

        # Create info
        info = self._create_info(current_price, reward)
        self.step_history.append(info)

        # Move to next step
        self.current_step += 1
        obs = self._get_obs()

        truncated = self.current_step >= len(self.df) - 1

        return obs, reward, terminated, truncated, info

    def _decode_action(self, raw_signal, position_size) -> tuple[int, float]:
        """Convert raw action to trading signal"""
        if raw_signal >= self.buy_threshold:
            return 1, position_size  # BUY
        elif raw_signal <= self.sell_threshold:
            return -1, position_size  # SELL
        else:
            return 0, 0.0  # HOLD

    def _calculate_reward(self, current_price: float) -> float:
        if self.last_broker_state is None:
            return 0.0  # No reward on first step

        return np.log(self.broker.equity / self.last_broker_state['equity'])

    """ def _calculate_reward(self, step_pnl, current_price):
        initial = self.broker.initial_balance
        cash_used = self.broker.cash_used
        cash = self.broker.cash
        position = self.broker.position_shares
        entry = self.broker.entry_price

        pct_change = (cash - initial) / initial

        # Base reward: proportional to equity growth
        reward = np.tanh(5 * pct_change)

        # If in position
        if abs(position) > 0:
            unrealized = self.broker.calculate_unrealized_pnl(current_price)
            reward += 0.003 * np.tanh(unrealized / cash_used)
            # Only reward holding if profitable
            if unrealized > 0:
                distance = abs(current_price - entry) / entry
                reward += 0.01 * distance

        # Penalize overtrading
        if self.broker.traded:
            reward -= 0.002

        # Hard clamp: never positive if equity < initial
        if cash < initial:
            reward = min(reward, 0)

        return float(np.clip(reward, -1, 1)) """

    def _update_volume_profiles(self):
        def safe_vp(df_slice):
            vp = get_vp_levels(df_slice['close'], df_slice['volume'])
            return {k: 0.0 if np.isnan(v) else v for k, v in vp.items()}

        date = self.df['date'][self.current_step]
        date = pd.Timestamp(date).floor('D')

        # Daily VP
        day_data = self.df[(self.df['date'].dt.floor('D') == date) & (self.df.index < self.current_step)]
        self.daily_vp = safe_vp(day_data)

        # Previous day VP
        prev_date = date - pd.Timedelta(days=1)
        prev_data = self.df[(self.df['date'].dt.floor('D') == prev_date) & (self.df.index < self.current_step)]
        self.prev_day_vp = safe_vp(prev_data) if len(prev_data) > 0 else self.daily_vp

        # Weekly VP (last 7 days)
        week_start = date - pd.Timedelta(days=6)
        week_data = self.df[(self.df['date'].dt.floor('D') >= week_start) & (self.df.index < self.current_step)]
        self.weekly_vp = safe_vp(week_data) if len(week_data) > 0 else self.daily_vp

    def _get_obs(self):
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step

        # Slice features and prices
        price_features = self.price_features[start_idx:end_idx].values
        indicator_features = self.indicator_features[start_idx:end_idx].values
        current_prices = self.df['close'].iloc[start_idx:end_idx].values

        # Update VP levels for current step (causal)
        self._update_volume_profiles()  # sets self.daily_vp, self.prev_day_vp, self.weekly_vp
        price = current_prices[-1]
        vp_daily_features = np.array([
            (price - self.daily_vp['vah']) / price,
            (price - self.daily_vp['poc']) / price,
            (price - self.daily_vp['val']) / price,
        ], dtype=np.float32)

        vp_prev_day_features = np.array([
            (price - self.prev_day_vp['vah']) / price,
            (price - self.prev_day_vp['poc']) / price,
            (price - self.prev_day_vp['val']) / price,
        ], dtype=np.float32)

        vp_week_features = np.array([
            (price - self.weekly_vp['vah']) / price,
            (price - self.weekly_vp['poc']) / price,
            (price - self.weekly_vp['val']) / price
        ], dtype=np.float32)

        # Return as dict instead of concatenation
        obs = {
            'price': price_features,
            'features': indicator_features,
            'volume_profile_week': vp_week_features,
            'volume_profile_prev_day': vp_prev_day_features,
            'volume_profile_daily': vp_daily_features,
            'account': self._get_account_obs(),
        }
        return obs

    def _get_account_obs(self):

        # Sliding window
        start_idx = max(0, self.current_step - self.window_size)
        steps = range(start_idx, self.current_step)

        # Broker states aligned with the window
        broker_states = []
        for i, step in enumerate(steps):
            history_index = len(self.broker.step_history) - (self.current_step - step)
            if 0 <= history_index < len(self.broker.step_history):
                state = self.broker.step_history[history_index]
            else:
                state = self.broker._create_step_record(step, 0.0, 0, 0.0, 0.0)
            broker_states.append(state)

        # Clamp feature access
        initial_balance = self.broker.initial_balance

        # Percentage normalization: 100% = 0.1, 200% = 0.2, etc.
        norm_factor = initial_balance * 10

        # Calculate account info
        position_direction = np.array([
            1.0 if s['cash_used'] > 0.0 else -1.0 if s['cash_used'] < 0.0 else 0.0
            for s in broker_states
        ], dtype=np.float32)

        position_size = np.array([s['position_size'] for s in broker_states], dtype=np.float32)

        cash = np.array([s['cash'] for s in broker_states], dtype=np.float32)
        position_value = np.array([s['cash_used'] for s in broker_states], dtype=np.float32)
        unrealized_pnl = np.array([s['unrealized_pnl'] for s in broker_states], dtype=np.float32)
        traded = np.array([1.0 if s['traded'] else 0.0 for s in broker_states], dtype=np.float32)

        # Assemble account info
        account_info = np.stack([
            position_direction,
            position_size,  # already normalized this is the value we received from the action
            cash / norm_factor,
            position_value / cash.clip(min=1e-8),
            unrealized_pnl / position_value.clip(min=1e-8),
            traded,
        ], axis=1)

        return account_info

    def _create_info(self, price, reward):
        """Create info dictionary with action validation feedback"""
        broker_state = self.broker.get_state()

        # Add action validation info
        return {
            'price': price,
            'reward': reward,
            **broker_state  # Include all broker state
        }
