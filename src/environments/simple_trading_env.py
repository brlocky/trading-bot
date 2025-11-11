"""
SimpleTradingEnv - Multi-Input Trading Environment (Multi-Scale Architecture)

Uses 6 feature groups with temporal/spatial separation for optimal learning:

1. Micro Temporal (5): OHLC + Volume [Small-kernel CNN]
   - Fine-grained price movements, temporal patterns
2. Micro Spatial (4): Body/wick ratios [MLP]
   - Candle structure, no temporal dependency
3. Meso Patterns (2): 1h, 4h returns [Medium-kernel CNN]
   - Intraday trends
4. Macro Patterns (1): 24h return [Large-kernel CNN]
   - Daily trends
5. Account State (5): Balance, equity, PnL [MLP]
   - Current account metrics
6. Position Info (7): Status, leverage, SL/TP distances [MLP]
   - Current position state

Total: 24 features per timestep (price patterns) + 12 features (trading state)
Architecture: Multi-scale CNN (3 scales) + GNN (optional) + 2 MLPs -> fused features

Action Space: MultiDiscrete([4]) - [Direction: HOLD/LONG/SHORT/CLOSE]
"""

from copy import deepcopy
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from gymnasium.spaces import MultiDiscrete
from environments import SimpleBroker
from data_processing.enhanced_features import (
    get_account_state_features,
    get_position_info_features,
    precompute_micro_temporal_features,
    precompute_micro_spatial_features,
    precompute_meso_patterns_features,
    precompute_macro_patterns_features,
)
from utils.PivotsSLTPCalculator import PivotSLTPCalculator
from utils.indicator_utils import add_indicators


class SimpleTradingEnv(gym.Env):
    """
    Multi-Input Trading Environment with Multi-Scale Feature Separation

    Separates features by temporal scale and processing type:
    - Temporal features (time-series) → CNN encoders
    - Spatial features (per-candle structure) → MLP encoder
    - Trading state (account/position) → MLP encoders
    """

    def __init__(
        self, data, initial_balance=10000, lookback_window=288,
        n_bins=50, device="cuda", render_mode='human',
        enable_pattern_memory=False
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
        intial_data_len = len(self.data)
        self.data = self.data.dropna().reset_index(drop=True)
        data_len_after_na = len(self.data)
        if data_len_after_na < intial_data_len:
            print(f"Info: Dropped {intial_data_len - data_len_after_na} rows due to NaNs after adding indicators.")

        self.data_len = len(self.data)

        # Initialize pattern memory
        self.enable_pattern_memory = enable_pattern_memory
        if enable_pattern_memory:
            from environments.trading_pattern_memory import TradingPatternMemory
            self.pattern_memory = TradingPatternMemory()

        # Track current episode
        self.episode_transitions = []
        self.episode_return = 0.0

        # Prepare data with all features - pre-compute all static features (modifies in place)
        self.micro_temporal_cols = precompute_micro_temporal_features(self.data, window=lookback_window)
        self.micro_spatial_cols = precompute_micro_spatial_features(self.data, window=lookback_window)
        self.meso_cols = precompute_meso_patterns_features(self.data, window=lookback_window)
        self.macro_cols = precompute_macro_patterns_features(self.data, window=lookback_window)

        # Pre-convert price features to tensors (avoid DataFrame→NumPy→Tensor conversion on every step)
        self.micro_temporal_tensor = torch.from_numpy(
            self.data[self.micro_temporal_cols].values.astype(np.float32)
        ).to(device)

        self.micro_spatial_tensor = torch.from_numpy(
            self.data[self.micro_spatial_cols].values.astype(np.float32)
        ).to(device)

        self.meso_tensor = torch.from_numpy(
            self.data[self.meso_cols].values.astype(np.float32)
        ).to(device)

        self.macro_tensor = torch.from_numpy(
            self.data[self.macro_cols].values.astype(np.float32)
        ).to(device)

        # Action space: [direction, risk_reward_ratio, atr_multiplier]
        # - direction: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
        # - risk_reward_ratio: 0-9 (maps to 1.0x-10.0x)
        # - atr_multiplier: 0-9 (maps to 1.0-2.8)
        # self.action_space = MultiDiscrete([4, 10, 10])
        self.action_space = MultiDiscrete([4])

        self.observation_space = gym.spaces.Dict({
            'micro_temporal': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.micro_temporal_cols)),
                dtype=np.float32
            ),
            'micro_spatial': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.micro_spatial_cols)),
                dtype=np.float32
            ),
            'meso_patterns': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.meso_cols)),
                dtype=np.float32
            ),
            'macro_patterns': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.macro_cols)),
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
        # self.vp = EnhancedVolumeProfile(n_bins=n_bins, lookback_window=lookback_window, device=device)

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
        # self.vp.reset()

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
        risk_reward_ratio = 2.5
        atr_multiplier = 2.0

        # Convert to actual prices
        current_price = self.data['close'].iloc[self.current_step].item()
        current_price_high = self.data['high'].iloc[self.current_step].item()
        current_price_low = self.data['low'].iloc[self.current_step].item()

        previous_state = self.broker.get_state()
        previous_trades = deepcopy(self.broker.trade_history)

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
        current_trades = self.broker.trade_history

        reward = self.calculate_reward(direction_action, current_trades, previous_trades, current_state, previous_state)

        # Track transition for pattern memory
        if self.enable_pattern_memory:
            self.episode_transitions.append({
                'step': self.current_step,
                'action': action,
                'reward': reward,           # Normalized reward
                'info': current_state.copy()
            })
            self.episode_return += reward

        done = False
        truncated = False
        if self.broker.is_bankrupt:
            print("Account bankrupt!")
            done = True
            # Bankruptcy penalty already applied in calculate_reward() - don't double penalize

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
        Generate multi-input observation with multi-scale feature groups.

        Groups:
        1. Micro Temporal (5): OHLC + Volume → Small kernel CNN
        2. Micro Spatial (4): Candle structure → MLP (last timestep)
        3. Meso Patterns (2): Intraday trends → Medium kernel CNN
        4. Macro Patterns (1): Daily trends → Large kernel CNN
        5. Account State (5): Current account metrics → MLP
        6. Position Info (7): Current position state → MLP
        """
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step

        # Group 1-4: Multi-scale price patterns (tensor slicing + zero-copy conversion)
        # Use contiguous() to ensure efficient memory layout after slicing
        micro_temporal = self.micro_temporal_tensor[start_idx:end_idx].contiguous()
        micro_spatial = self.micro_spatial_tensor[start_idx:end_idx].contiguous()
        meso_patterns = self.meso_tensor[start_idx:end_idx].contiguous()
        macro_patterns = self.macro_tensor[start_idx:end_idx].contiguous()

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

        # Build observation dictionary (will be converted to numpy arrays below)
        obs_dict: dict = {  # type: ignore - will contain numpy arrays after conversion
            'micro_temporal': micro_temporal,
            'micro_spatial': micro_spatial,
            'meso_patterns': meso_patterns,
            'macro_patterns': macro_patterns,
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

            # Convert to NumPy array for Stable-Baselines3 (minimize copies)
            if tensor.is_cuda:
                obs_dict[key] = tensor.cpu().numpy().astype(np.float32)
            else:
                obs_dict[key] = tensor.numpy().astype(np.float32)

        return obs_dict

    def calculate_reward(self, action, current_trades, previous_trades, current_state, previous_state):
        """
        REWARD FUNCTION v30 - AGGRESSIVE EXPLORATION + SHAPED REWARDS

        Changes from v29:
        - Increased exploration bonus: 0.1 → 0.3
        - Added position holding reward: +0.01/step when in position
        - Increased balance change scaling: 30x → 100x
        - Reduced invalid action penalties
        """
        if previous_state is None:
            return 0.0

        reward = 0.0
        current_balance = current_state.get('equity', 0)
        previous_balance = previous_state.get('equity', 0)
        current_position = current_state.get('position_size', 0)
        previous_position = previous_state.get('position_size', 0)

        # === 1. BALANCE CHANGE (AMPLIFIED!) ===
        balance_change = current_balance - previous_balance
        balance_change_pct = balance_change / self.initial_balance

        # INCREASED: Scale balance change more aggressively
        # Typical 1% change → ±1.0 reward (was ±0.3)
        balance_reward = balance_change_pct * 100.0
        balance_reward = float(np.clip(balance_reward, -1.0, 1.0))
        reward += balance_reward

        # === 2. POSITION HOLDING REWARD ===
        # Reward for being in a position (encourages action)
        if current_position != 0 and previous_position != 0:
            reward += 0.1  # Small reward per step in position

        # === 3. TRADE COMPLETION REWARDS ===
        current_closed = [t for t in current_trades if t.get('status') == 'CLOSED']
        previous_closed = [t for t in previous_trades if t.get('status') == 'CLOSED']

        if len(current_closed) > len(previous_closed):
            new_trade = current_closed[-1]
            reason = new_trade.get('reason', 'Unknown')

            if 'TP' in reason:
                reward += 1.0  # Maximum reward

            elif 'SL' in reason:
                reward += -0.5  # 2:1 ratio

            else:  # Manual close
                reward += -0.05

        # === 4. EXPLORATION INCENTIVES (INCREASED!) ===
        # Opening new position bonus - TRIPLED!
        if previous_position == 0 and current_position != 0:
            reward += 0.3  # Was 0.1

        # === 5. REDUCED INVALID ACTION PENALTIES ===
        # Very small penalties - don't discourage exploration
        if action == 3 and previous_position == 0:
            reward -= 0.1  # Was 0.05

        if action in [1, 2] and previous_position != 0:
            reward -= 0.1  # Was 0.1

        # === 6. BANKRUPTCY PENALTY ===
        if self.broker.is_bankrupt:
            reward = -1.0

        return reward

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
