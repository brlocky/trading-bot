"""
SimpleTradingEnv - Multi-Input Trading Environment (Optimized Architecture)

Uses 6 feature groups with semantic separation for optimal learning:

1. Price Patterns (8): Candle structure, volume, multi-TF returns [CNN]
   - Temporal sequences processed by Conv1d for pattern detection
2. Market Context (6): EMA/VWAP distances, volatility [MLP]
   - Current market positioning (last timestep only)
3. Trend Indicators (10): EMA slopes, crossovers, momentum [MLP]
   - Current trend state (last timestep only)
4. Trading Sessions (3): Asia/London/NY flags [Linear]
   - Current session (last timestep only)
5. Account State (5): Balance, equity, PnL, commission [MLP]
   - Current account metrics
6. Position Info (7): Status, leverage, distances, risk-reward [MLP]
   - Current position state

Total: 39 features per timestep
Architecture: 1 CNN encoder + 5 MLP/Linear encoders -> 256-dim fused features

Action Space: MultiDiscrete([4, 10, 10]) - [Direction, Risk-Reward Ratio, ATR Multiplier]
"""

import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from gymnasium.spaces import MultiDiscrete
from environments import SimpleBroker
from environments.generic_trading_visualizer import GenericTradingVisualizer, create_advanced_config
from features.enhanced_volume_profile import EnhancedVolumeProfile
from data_processing.enhanced_features import (
    get_account_state_features,
    get_position_info_features,
    precompute_price_patterns_features,
    precompute_market_context_features,
    precompute_trend_features,
    precompute_trading_sessions
)
from utils.PivotsSLTPCalculator import PivotSLTPCalculator

from utils.indicator_utils import add_indicators


class SimpleTradingEnv(gym.Env):
    """
    Multi-Input Trading Environment (Phase 2: Divergence Detection)

    Dual CNN perspectives + divergence CNNs + minimal high-signal features.
    Total: 134 features + 50 VP bins = 184 features
    """

    def __init__(
        self, data, initial_balance=10000, lookback_window=288,
        n_bins=50, device="cuda", render_mode='human',
        enable_pattern_memory=True, reward_min=-200.0, reward_max=200.0
    ):
        super().__init__()

        self.lookback_window = lookback_window
        self.device = device
        self.initial_balance = initial_balance
        self.n_bins = n_bins
        self.render_mode = render_mode

        self.data = data.copy()
        # Add technical indicators if not already present
        self.data['date'] = pd.to_datetime(self.data['date_close'])
        add_indicators(self.data)
        self.data = self.data.dropna().reset_index(drop=True)

        # Reward normalization bounds (min-max scaling to [-1, 1])
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.reward_range_size = reward_max - reward_min

        # Initialize pattern memory
        self.enable_pattern_memory = enable_pattern_memory
        if enable_pattern_memory:
            from environments.trading_pattern_memory import TradingPatternMemory
            self.pattern_memory = TradingPatternMemory()

        # Track current episode
        self.episode_transitions = []
        self.episode_return = 0.0

        # Prepare data with all features - pre-compute all static features (modifies in place)
        self.price_patterns_cols = precompute_price_patterns_features(self.data, window=lookback_window)
        self.market_context_cols = precompute_market_context_features(self.data, window=lookback_window)
        self.trend_feature_cols = precompute_trend_features(self.data)
        self.trading_session_cols = precompute_trading_sessions(self.data)

        # Action space: [direction, risk_reward_ratio, atr_multiplier]
        # - direction: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
        # - risk_reward_ratio: 0-9 (maps to 1.0x-10.0x)
        # - atr_multiplier: 0-9 (maps to 1.0-2.8)
        # self.action_space = MultiDiscrete([4, 10, 10])
        self.action_space = MultiDiscrete([4])

        self.observation_space = gym.spaces.Dict({
            'price_patterns': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.price_patterns_cols)),
                dtype=np.float32
            ),
            'market_context': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.market_context_cols)),
                dtype=np.float32
            ),
            'trend_indicators': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.trend_feature_cols)),
                dtype=np.float32
            ),
            'trading_sessions': gym.spaces.Box(
                low=0, high=1,
                shape=(lookback_window, len(self.trading_session_cols)),
                dtype=np.float32
            ),
            'account_state': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 5),
                dtype=np.float32
            ),
            'position_info': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 7),
                dtype=np.float32
            ),
        })

        # Initialize broker, volume profile, and zigzag state
        self.broker = SimpleBroker(initial_balance=self.initial_balance, quantity_precision=0.001)
        self.vp = EnhancedVolumeProfile(n_bins=n_bins, lookback_window=lookback_window, device=device)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Save previous episode if it exists
        if self.enable_pattern_memory and len(self.episode_transitions) > 0:
            episode_data = self._create_episode_summary()
            self.pattern_memory.add_episode(episode_data)

        # Reset episode tracking
        self.episode_transitions = []
        self.episode_return = 0.0

        self.current_step = self.lookback_window
        self.history = []
        self.broker.reset()
        self.vp.reset()

        # Initialize reward tracking
        self.steps_since_close = 0

        # Warm up the broker and volume profile
        for i in range(self.lookback_window):
            # Initialize broker state
            self.broker.step(i, 0, self.data['close'].iloc[i], self.data['high'].iloc[i], self.data['low'].iloc[i], 0, 0)

            # Record initial state
            self.history.append({
                "step": i,
                "action": 0,
                "reward": 0.0,
                "done": False,
                "truncated": False,
                **self.broker.get_state()
            })

        return self._get_obs(), {}

    def step(self, action):

        # Unpack action: [direction, risk_reward_idx, atr_idx]
        # direction_action, rr_idx, atr_idx = int(action[0]), int(action[1]), int(action[2])
        # risk_reward_ratio = 1.0 + (rr_idx * 1.0)  # Maps 0-9 to 1.0-10.0
        # tr_multiplier = 2.0 + (atr_idx * 0.3)    # Maps 0-9 to 2.0-4.7 (wider SL)
        direction_action = int(action[0])
        risk_reward_ratio = 5
        atr_multiplier = 2

        # Convert to actual prices
        current_price = self.data['close'].iloc[self.current_step].item()
        current_price_high = self.data['high'].iloc[self.current_step].item()
        current_price_low = self.data['low'].iloc[self.current_step].item()

        previous_state = self.broker.get_state()

        # Calculate SL/TP only for LONG and SHORT actions
        sl_price, tp_price = PivotSLTPCalculator.calculate_sl_tp(
            data=self.data,
            current_step=self.current_step,
            entry_price=current_price,
            direction=direction_action,
            risk_reward_ratio=risk_reward_ratio,  # Use agent's choice
            atr_multiplier=atr_multiplier         # Use agent's choice
        ) if direction_action in [1, 2] else (None, None)

        # Pass action directly to broker - it handles the logic
        # Actions: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
        self.broker.step(
            step_index=self.current_step,
            signal=direction_action,
            close=current_price,
            high=current_price_high,
            low=current_price_low,
            tp_price=tp_price,
            sl_price=sl_price
        )

        current_state = self.broker.get_state()

        reward = self.calculate_reward(direction_action, current_state, previous_state)

        # Normalize reward using min-max scaling to [-1, 1]
        raw_reward = reward
        reward = self._normalize_reward_minmax(reward)

        # Track transition for pattern memory
        if self.enable_pattern_memory:
            self.episode_transitions.append({
                'step': self.current_step,
                'action': action,
                'reward': reward,           # Normalized reward
                'raw_reward': raw_reward,   # Keep raw for analysis
                'info': current_state.copy()
            })
            self.episode_return += reward

        done = False
        truncated = False        # More lenient bankruptcy check - only stop if truly bankrupt (< 5% of initial)
        if self.broker.is_bankrupt:
            print("Account bankrupt!")
            done = True
            reward -= 100.0  # Large penalty for bankruptcy

        self.history.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            **self.broker.get_state()
        })

        self.current_step += 1
        truncated = self.current_step >= len(self.data) - 1 if truncated is False else truncated

        # If episode ends, save it
        if (done or truncated) and self.enable_pattern_memory and len(self.episode_transitions) > 0:
            episode_data = self._create_episode_summary()
            self.pattern_memory.add_episode(episode_data)
            self.episode_transitions = []
            self.episode_return = 0.0

        obs = self._get_obs()
        return obs, reward, done, truncated, self.history[-1]

    def _get_obs(self):
        """
        Generate multi-input observation with 6 feature groups.

        Groups:
        1. Price Patterns (8): Temporal sequences for CNN
        2. Market Context (6): Current positioning (last timestep)
        3. Trend Indicators (10): Current trend state (last timestep)
        4. Trading Sessions (3): Current session (last timestep)
        5. Account State (5): Current account metrics
        6. Position Info (7): Current position state
        """
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step

        # Get data slice
        df_slice = self.data.iloc[start_idx:end_idx].copy()

        # Group 1: Price Patterns (for CNN temporal processing)
        price_patterns = df_slice[self.price_patterns_cols].values
        price_patterns = torch.tensor(price_patterns, dtype=torch.float32, device=self.device)

        # Group 2: Market Context (current positioning)
        market_context = df_slice[self.market_context_cols].values
        market_context = torch.tensor(market_context, dtype=torch.float32, device=self.device)

        # Group 3: Trend Indicators
        trend_indicators = df_slice[self.trend_feature_cols].values
        trend_indicators = torch.tensor(trend_indicators, dtype=torch.float32, device=self.device)

        # Group 4: Trading Sessions
        trading_sessions = df_slice[self.trading_session_cols].values
        trading_sessions = torch.tensor(trading_sessions, dtype=torch.float32, device=self.device)

        # Group 5: Account State
        account_state = get_account_state_features(
            self.broker.step_history,
            self.initial_balance,
            self.lookback_window
        ).to(self.device)

        # Group 6: Position Info
        position_info = get_position_info_features(
            self.broker.step_history,
            self.lookback_window,
        ).to(self.device)

        # Build observation dictionary
        obs_dict = {
            'price_patterns': price_patterns,
            'market_context': market_context,
            'trend_indicators': trend_indicators,
            'trading_sessions': trading_sessions,
            'account_state': account_state,
            'position_info': position_info,
        }

        # Sanitize all tensors and convert to NumPy for Stable-Baselines3
        for key, tensor in obs_dict.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: NaN/Inf in {key} at step {self.current_step}")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
                raise ValueError(f"NaN/Inf detected in observation tensor '{key}'.")

            # Check for values outside [-1, 1] range
            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()
            if tensor_min < -1.0 or tensor_max > 1.0:
                # Find indices of invalid values
                min_idx = (tensor == tensor_min).nonzero(as_tuple=False)[0].tolist()
                max_idx = (tensor == tensor_max).nonzero(as_tuple=False)[0].tolist()
                print(f"Warning: Values outside [-1, 1] in '{key}' at step {self.current_step}:")
                print(f"  min={tensor_min:.4f} at indices {min_idx[:5]}{'...' if len(min_idx) > 5 else ''}")
                print(f"  max={tensor_max:.4f} at indices {max_idx[:5]}{'...' if len(max_idx) > 5 else ''}")

            # Convert to CPU NumPy array for Stable-Baselines3
            obs_dict[key] = tensor.cpu().numpy().astype(np.float32)

        return obs_dict

    def _normalize_reward_minmax(self, reward):
        """
        Normalize reward to [-1, 1] using symmetric scaling around zero

        This ensures 0 reward → 0 normalized (centered)
        Positive rewards → [0, 1]
        Negative rewards → [-1, 0]

        Args:
            reward: Raw reward value

        Returns:
            Normalized reward in [-1, 1] range
        """
        # Handle edge case
        if reward == 0:
            return 0.0

        # Normalize positive and negative rewards separately (symmetric around 0)
        if reward > 0:
            # Positive: scale [0, reward_max] → [0, 1]
            normalized = reward / self.reward_max
        else:
            # Negative: scale [reward_min, 0] → [-1, 0]
            normalized = reward / abs(self.reward_min)

        # Clip to ensure bounds (handles extreme outliers)
        return float(np.clip(normalized, -1.0, 1.0))

    def calculate_reward(self, direction, current_state, previous_state):
        """
        REWARD FUNCTION v24 - ENCOURAGING ACTIVE TRADING

        Key Changes from v23:
        - INCREASED exploration bonus: 2.0 → 10.0 (encourage trading)
        - REDUCED SL penalties: -15-30 → -10-20 (less fear of losses)
        - ADDED idle penalty: -0.5 per step when flat (discourage excessive HOLDing)
        - Kept balance-focused rewards and TP bonuses
        - Kept redundant action penalty (-50)
        """
        if previous_state is None:
            return 0.0

        reward = 0.0

        # === 0. REDUNDANT ACTION PENALTY (CRITICAL FOR DISCRETE BOT) ===
        # Penalize trying to open LONG when already LONG, or SHORT when already SHORT
        previous_position = previous_state.get('position_size', 0)
        current_position = current_state.get('position_size', 0)

        # Check if position didn't change (means action was ignored by broker)
        if previous_position != 0 and current_position == previous_position:
            # Get the last action from history
            if direction != 0:  # IF not HOLD
                # HARD PENALTY: Agent should learn to use HOLD or CLOSE instead
                reward -= 50.0

        # === 1. BALANCE CHANGE (PRIMARY GOAL) ===
        # Track balance (not equity) to reward only REALIZED gains
        current_balance = current_state.get('current_balance')
        previous_balance = previous_state.get('current_balance')
        balance_change = current_balance - previous_balance

        # Strong multiplier - balance growth is the main objective
        if balance_change != 0:
            reward += balance_change * 0.5  # 50% of dollar change as reward

        # === 1. TRADE COMPLETION REWARDS (SECONDARY SIGNAL) ===
        current_trades = current_state.get('trades', [])
        previous_trades = previous_state.get('trades', [])
        current_closed = [t for t in current_trades if t.get('status') == 'CLOSED']
        previous_closed = [t for t in previous_trades if t.get('status') == 'CLOSED']

        if len(current_closed) > len(previous_closed):
            new_trade = current_closed[-1]
            pnl_percent = new_trade.get('pnl_percent', 0)
            duration = new_trade.get('duration', 0)
            reason = new_trade.get('reason', 'Unknown')

            # Note: Balance change already rewarded above, this adds trade-specific bonuses

            if 'TP' in reason:
                # Big reward for TP hits (scales with profit)
                base_reward = 50.0 + abs(pnl_percent) * 10.0

                # Duration bonus: FAST TP = EXCELLENT! Slow TP = less capital efficient
                if duration <= 10:
                    duration_bonus = 1.5  # BONUS for quick wins (1-10 bars)
                elif duration <= 50:
                    duration_bonus = 1.2  # Good for medium-term (11-50 bars)
                elif duration <= 100:
                    duration_bonus = 1.0  # Neutral (51-100 bars)
                else:
                    duration_bonus = 0.8  # Slight penalty for very slow TPs (>100 bars)

                reward += base_reward * duration_bonus

            elif 'SL' in reason:
                # Reduced penalty for SL (was 15-30, now 10-20)
                penalty = 10.0 + abs(pnl_percent) * 2.0  # Reduced multiplier from 3.0

                # Extra penalty for very quick losses
                if duration < 5:
                    penalty *= 1.5  # Hit SL too fast = bad trade

                reward -= penalty

            else:  # Manual close
                # Penalize manual closes unless profitable
                if pnl_percent > 1.0:  # Only reward if >1% profit
                    reward += 10.0 + pnl_percent * 5.0
                elif pnl_percent > 0:
                    reward += 2.0  # Small reward for small profit
                else:
                    # Penalty for closing at loss
                    reward -= 10.0 + abs(pnl_percent) * 2.0

        # === 2. EXPLORATION BONUS (Increased to encourage trading) ===
        current_position = current_state.get('position_size', 0)
        previous_position = previous_state.get('position_size', 0)

        if previous_position == 0 and current_position != 0:
            reward += 10.0  # Increased from 2.0 - significant reward for taking action

        # === 2b. IDLE PENALTY (Discourage excessive HOLDing) ===
        if current_position == 0 and previous_position == 0:
            reward -= 0.5  # Small penalty for staying flat

        # === 3. POSITION MANAGEMENT (Encourage holding winners) ===
        if current_position != 0:
            unrealized_pnl = current_state.get('unrealized_pnl', 0)
            used_balance = current_state.get('used_balance', 1)
            unrealized_pct = (unrealized_pnl / used_balance * 100) if used_balance > 0 else 0

            # Reward holding winning positions
            if unrealized_pct > 1.0:
                reward += min(unrealized_pct * 0.3, 3.0)  # Small capped reward

            # Penalize deep drawdowns
            elif unrealized_pct < -3.0:
                reward += unrealized_pct * 0.3  # Negative value

            # Between -3% and +1%: NEUTRAL

        else:
            # Track inaction when flat
            self.steps_since_close += 1

            # Penalize excessive inaction (>500 steps = ~42 hours)
            if self.steps_since_close > 500:
                inaction_penalty = min((self.steps_since_close - 500) * 0.02, 15.0)
                reward -= inaction_penalty

        # === 4. BANKRUPTCY PENALTY ===
        if self.broker.is_bankrupt:
            reward = -500.0

        # Clip to prevent extreme values
        return float(np.clip(reward, -500.0, 500.0))

    def _create_episode_summary(self):
        """Create summary of episode for pattern memory"""
        trades = [t for t in self.broker.step_history if t.get('status') == 'CLOSED']
        winning_trades = [t for t in trades if t.get('pnl_percent', 0) > 0]

        # Calculate market conditions from recent data
        recent_window = 20
        end_idx = min(self.current_step, len(self.data) - 1)
        start_idx = max(0, end_idx - recent_window)

        return {
            'transitions': self.episode_transitions.copy(),
            'total_return': self.episode_return,
            'total_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0.0,
            'final_balance': self.broker.get_state()['current_balance'],
            'episode_length': len(self.episode_transitions),
            'market_conditions': {
                'volatility': float(self.data['close'].iloc[start_idx:end_idx].std()),
                'avg_volume': float(self.data['volume'].iloc[start_idx:end_idx].mean()),
            }
        }

    def render(self):
        visualizer = GenericTradingVisualizer(subplot_config=create_advanced_config())

        # Prepare indicators - just pass the volume profile object
        indicators = {
            'volume_profile': self.vp,  # Pass the entire EnhancedVolumeProfile object
            'history_data': getattr(self, 'history', [])
        }

        frame = visualizer.plot_data(
            data=self.data,
            trade_history=self.broker.step_history,
            indicators=indicators,
            current_step=self.current_step,
            lookback_window=getattr(self, 'lookback_window', 100),
            title=f"Trading Environment - Step {self.current_step}"
        )
        return frame
