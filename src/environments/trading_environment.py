"""
Trading Environment for Reinforcement Learning

Standalone trading environment that can be used with any RL framework.
Uses separate normalization layer for proper data handling.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

from data_processing.feature_normalizer import FeatureNormalizer


class TradingEnvironment(gym.Env):
    """
    Trading environment for RL agent with proper normalization separation

    State: window of normalized features + account info
    Action: [-1, 1] position size (short/flat/long)
    Reward: PnL change minus drawdown penalty
    """

    def __init__(self,
                 df: pd.DataFrame,
                 feature_config: Dict[str, str],
                 initial_balance: float = 1000000.0,
                 window_size: int = 672,
                 fit_normalizer: bool = True):
        """
        Initialize trading environment with dual action space:
        - Action[0]: Signal [-1, 1] (sell/hold/buy)  
        - Action[1]: Position Size [0, 1] (fraction of account)

        Args:
            df: DataFrame with OHLCV data and features
            feature_config: Dict mapping feature names to normalization types
            initial_balance: Starting balance
            window_size: Observation window size
            fit_normalizer: Whether to fit normalizer to this data (False for test sets)
        """
        super().__init__()

        # Store parameters
        self.initial_balance = initial_balance
        self.window_size = window_size

        # Store original data
        self.df = df.copy()

        # Store Feature configuration
        self.feature_config = feature_config
        self.feature_columns = list(feature_config.keys())

        # Validate features exist in dataframe
        missing_features = set(self.feature_columns) - set(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing features in DataFrame: {missing_features}")

        # Configure action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.001]),   # [signal_min, position_size_min]
            high=np.array([1.0, 1.0]),     # [signal_max, position_size_max]
            shape=(2,),                    # Two outputs
            dtype=np.float32
        )

        # Features + [Balance Change, Position, Last Signal, Last Position Size, Total Return %, Unrealized PnL, Position Exposure, Step PnL]
        obs_dim = len(self.feature_columns) + 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, obs_dim),
            dtype=np.float32
        )

        try:
            # Setup normalization
            self._setup_normalizer(fit_normalizer)

            # Normalize the data
            self._prepare_normalized_data()

            # Initialize environment state
            self.reset()
        except Exception as e:
            print(f"Error during environment initialization: {e}")
            raise e

    def _setup_normalizer(self, fit_normalizer: bool):
        # Create new normalizer
        self.normalizer = FeatureNormalizer(self.feature_config)

        if fit_normalizer:
            print("🔧 Fitting new normalizer to training data...")
            self.normalizer.fit(self.df[self.feature_columns], close_prices=self.df['close'])
        else:
            raise ValueError("Cannot create unfitted normalizer without fit_normalizer=True")

    def _prepare_normalized_data(self):
        """Pre-normalize all features for efficient observation retrieval"""
        print("⚡ Pre-normalizing feature data...")

        # Normalize features
        normalized_features = self.normalizer.transform(
            self.df[self.feature_columns],
            close_prices=self.df['close']
        )

        # Store normalized features as numpy array for fast access
        self.normalized_features = normalized_features[self.feature_columns].values

        print(f"✅ Pre-normalized {len(self.feature_columns)} features")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset with dual action tracking"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.done = False

        # Initialize dual action tracking
        self.last_signal = 0.0
        self.last_position_size = 0.0

        # Initialize tracking metrics
        self.step_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.position_exposure = 0.0

        if seed is not None:
            np.random.seed(seed)

        return self._get_obs(), {}

    def _get_obs(self):
        """Updated observation with dual action feedback"""
        # Get the window of normalized features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        feature_window = self.normalized_features[start_idx:end_idx]

        # Enhanced account info with dual action history
        account_info = np.array([
            (self.balance / self.initial_balance - 1.0),  # Balance change
            np.tanh(self.position),                       # Current position
            getattr(self, 'last_signal', 0.0),          # Previous signal [-1, 1]
            getattr(self, 'last_position_size', 0.0),   # Previous position size [0, 1]

            # Trading metrics (same as before)
            ((self.balance / self.initial_balance) - 1) * 100,  # Total return %
            self.unrealized_pnl / self.initial_balance,          # Normalized unrealized PnL
            self.position_exposure / self.initial_balance,       # Normalized position exposure
            self.step_pnl / self.initial_balance                 # Normalized step PnL
        ])

        # Broadcast account info to match window size
        account_window = np.tile(account_info, (self.window_size, 1))

        # Concatenate features and account info
        obs = np.concatenate([feature_window, account_window], axis=1)

        # Final safety check for NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)

        return obs.astype(np.float32)

    def step(self, action):
        """Execute one step with additive position management"""
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Store previous state
        prev_balance = self.balance
        prev_position = self.position

        # Parse dual actions
        signal = float(action[0])      # [-1, 1]: sell/hold/buy direction
        position_size = float(action[1])  # [0.001, 1]: fraction of account
        price = self.df.iloc[self.current_step]['close']

        # Store actions for next observation
        self.last_signal = signal
        self.last_position_size = position_size

        # Apply thresholds to filter out weak signals
        buy_threshold = 0.1
        sell_threshold = -0.1

        # Calculate target position (not replacement position)
        target_position = 0.0
        if signal > buy_threshold:
            # LONG: Target position size
            target_position = position_size  # e.g., 0.8 = 80% of account
        elif signal < sell_threshold:
            # SHORT: Target negative position
            target_position = -position_size  # e.g., -0.3 = 30% short
        else:
            # HOLD: Target zero position
            target_position = 0.0

        # Calculate position change (incremental trade)
        position_change = target_position - self.position

        # Calculate reward and trade metrics
        reward = 0.0
        step_pnl = 0.0
        trade_occurred = abs(position_change) > 0.001  # Small threshold for precision

        # Execute the position change
        if trade_occurred:
            # Close/adjust existing position if needed
            if self.position != 0:
                # PnL from existing position
                existing_pnl = (price - self.entry_price) * self.position
                self.balance += existing_pnl
                step_pnl += existing_pnl

            # Update to target position
            self.position = target_position

            # Set new entry price for the target position
            if self.position != 0:
                self.entry_price = price

            # Calculate reward from the trade
            reward = step_pnl / self.initial_balance

        # Update tracking metrics
        self.step_pnl = step_pnl
        self.unrealized_pnl = 0.0 if self.position == 0 else (price - self.entry_price) * self.position
        self.position_exposure = abs(self.position) * price

        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True

        obs = self._get_obs()

        # Enhanced info with position change tracking
        info = {
            # Core state
            'balance': self.balance,
            'position': self.position,
            'price': price,
            'step': self.current_step,

            # PnL tracking
            'step_pnl': step_pnl,
            'total_pnl': self.balance - self.initial_balance,
            'total_return_pct': ((self.balance / self.initial_balance) - 1) * 100,
            'unrealized_pnl': self.unrealized_pnl,

            # Enhanced position tracking
            'signal_raw': signal,
            'position_size_raw': position_size,
            'target_position': target_position,      # What model wanted
            'position_change': position_change,      # Actual change made
            'trade_occurred': trade_occurred,
            'prev_position': prev_position,          # Previous position

            # Portfolio metrics
            'portfolio_value': self.balance,
            'reward': reward,
            'position_exposure': self.position_exposure,
            'entry_price': self.entry_price,
            'balance_change': self.balance - prev_balance,

            # Timestamp
            'timestamp': self.df.iloc[self.current_step-1]['time'] if 'time' in self.df.columns and self.current_step > 0 else None,
        }

        return obs, reward, self.done, False, info

    def get_normalizer(self) -> FeatureNormalizer:
        """Get the fitted normalizer for use with test environments"""
        return self.normalizer

    def save_normalizer(self, filepath: str):
        """Save the fitted normalizer"""
        self.normalizer.save(filepath)

    def get_feature_stats(self) -> Dict[str, Any]:
        """Get normalization statistics for inspection"""
        return self.normalizer.get_feature_stats()

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and normalization quality"""
        stats = {}

        # Check for NaN values in original features
        nan_counts = self.df[self.feature_columns].isnull().sum()
        stats['original_nan_counts'] = nan_counts[nan_counts > 0].to_dict()

        # Check normalized feature ranges
        normalized_stats = {}
        for i, feature in enumerate(self.feature_columns):
            feature_data = self.normalized_features[:, i]
            normalized_stats[feature] = {
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'nan_count': int(np.sum(np.isnan(feature_data)))
            }

        stats['normalized_features'] = normalized_stats

        return stats
