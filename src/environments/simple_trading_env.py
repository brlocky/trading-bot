"""
SimpleTradingEnv - Multi-Input Trading Environment (Multi-Scale Architecture)

Uses 8 feature groups with temporal/spatial separation for optimal learning:

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
7. VP Bins (n_bins): Volume distribution histogram [CNN]
   - Calculated from VISIBLE RANGE (lookback window)
   - Direct price-to-bin mapping for agent
8. VP Levels (9): 5 continuous + 4 binary features [Split MLP]
   - Continuous: high/vah/poc/val/low distances (ordered HIGH→LOW)
   - Binary: in_va, above_poc, session_intersection, poc_crossover
   - Calculated from VISIBLE RANGE

Total: 24 features (price patterns) + 12 (trading state) + n_bins + 9 (volume profile)
Architecture: Multi-scale CNN (4 scales) + 3 MLPs -> fused features

Action Space: MultiDiscrete([4]) - [Direction: HOLD/LONG/SHORT/CLOSE]

Volume Profile Notes:
- Uses Visible Range VP (not session-based)
- VP bins/levels calculated on-the-fly from lookback window data
- No session detection or circular buffers
- Agent can directly map VP bins to visible prices
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import MultiDiscrete
from environments import SimpleBroker
from data_processing.enhanced_features import (
    get_account_state_features,
    get_position_info_features,
    get_vp_bins_features_visible,
    get_vp_levels_features_visible,
    precompute_micro_temporal_features,
    precompute_micro_spatial_features,
    precompute_meso_patterns_features,
    precompute_macro_patterns_features,
)
from utils.PivotsSLTPCalculator import PivotSLTPCalculator
from utils.indicator_utils import add_indicators
from environments.trade_logger import TradeLogger


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
        n_bins=255, render_mode='human',
        maker_commission=0.00055,
        taker_commission=0.0002,
        vp_cache=None,
        enable_trade_logging=False,
        trade_log_dir="logs/trades"
    ):
        super().__init__()

        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.n_bins = n_bins
        self.render_mode = render_mode

        # Shared VP cache across all environments (passed from notebook)
        # Format: {step: {'bins': np.ndarray, 'levels': np.ndarray}}
        self.vp_cache = vp_cache if vp_cache is not None else {}

        # Trade logging for post-training analysis
        self.enable_trade_logging = enable_trade_logging
        self.trade_logger = TradeLogger(log_dir=trade_log_dir) if enable_trade_logging else None

        # Prepare data
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date_close'])
        add_indicators(self.data)
        initial_data_len = len(self.data)
        self.data = self.data.dropna().reset_index(drop=True)
        data_len_after_na = len(self.data)
        if data_len_after_na < initial_data_len:
            print(f"Info: Dropped {initial_data_len - data_len_after_na} rows due to NaNs after adding indicators.")

        self.data_len = len(self.data)

        # Prepare data with all features - pre-compute all static features (modifies in place)
        self.micro_temporal_cols = precompute_micro_temporal_features(self.data, window=lookback_window)
        self.micro_spatial_cols = precompute_micro_spatial_features(self.data, window=lookback_window)
        self.meso_cols = precompute_meso_patterns_features(self.data, window=lookback_window)
        self.macro_cols = precompute_macro_patterns_features(self.data, window=lookback_window)

        # Pre-convert price features to tensors (avoid DataFrame→NumPy→Tensor conversion on every step)
        self.micro_temporal_data = self.data[self.micro_temporal_cols].values.astype(np.float32)
        self.micro_spatial_data = self.data[self.micro_spatial_cols].values.astype(np.float32)
        self.meso_data = self.data[self.meso_cols].values.astype(np.float32)
        self.macro_data = self.data[self.macro_cols].values.astype(np.float32)

        # Action space: [direction, risk_reward_ratio, atr_multiplier]
        # - direction: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
        # - risk_reward_ratio: 0-9 (maps to 1.0x-10.0x)
        # - atr_multiplier: 0-9 (maps to 1.0-2.8)
        # self.action_space = MultiDiscrete([4, 10, 10])
        self.action_space = MultiDiscrete([3])  # [HOLD, LONG, SHORT] - CLOSE handled internally

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
            'vp_bins': gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(lookback_window, n_bins),
                dtype=np.float32
            ),
            'vp_levels': gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(lookback_window, 26),  # 20 continuous (OHLC×5) + 6 binary
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
        self.broker = SimpleBroker(
            initial_balance=self.initial_balance,
            quantity_precision=0.001,
            maker_commission=maker_commission,
            taker_commission=taker_commission
        )
        # self.vp = EnhancedVolumeProfile(n_bins=n_bins, lookback_window=lookback_window, device=device)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.history = []
        self.broker.reset()

        self.last_open_traded_step = -1

        # Warm up the broker with historical data
        for i in range(self.lookback_window):
            new_state = self.data.iloc[i]

            # Initialize broker state
            self.broker.step(i, 0, new_state, tp_price=None, sl_price=None)

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

    def get_action_mask(self):
        """Returns which actions are valid in current state"""
        if self.broker.position_size == 0:
            # Flat position - can open long, short, or hold
            return [True, True, True]  # [hold, long, short]
        else:
            # In position - can only close or hold
            return [True, False, False]  # [hold, long, short]

    def step(self, action):
        # Unpack action: [direction, risk_reward_idx, atr_idx]
        # direction_action, rr_idx, atr_idx = int(action[0]), int(action[1]), int(action[2])
        # risk_reward_ratio = 1.0 + (rr_idx * 1.0)  # Maps 0-9 to 1.0-10.0
        # tr_multiplier = 2.0 + (atr_idx * 0.3)    # Maps 0-9 to 2.0-4.7 (wider SL)
        action = int(action[0])
        risk_reward_ratio = 3
        atr_multiplier = 1.5

        current_mask = self.get_action_mask()
        if not current_mask[action]:
            # INVALID ACTION - stay in same market state, penalize
            reward = -10
            done = False
            truncated = False
            obs = self._get_obs()
            return obs, reward, done, truncated, self.history[-1]

        broker_previous_state = self.broker.get_state()

        # Calculate SL/TP only for LONG and SHORT actions
        # Calculate before update the step so that we dont leak future data, use current step
        sl_price, tp_price = None, None
        if action in [1, 2]:
            sl_price, tp_price = PivotSLTPCalculator.calculate_sl_tp(
                data=self.data,
                current_step=self.current_step,
                entry_price=self.data.iloc[self.current_step]['close'],
                direction=action,
                risk_reward_ratio=risk_reward_ratio,  # Use agent's choice
                atr_multiplier=atr_multiplier         # Use agent's choice
            )

        # Increase next step
        self.current_step += 1
        next_candle = self.data.iloc[self.current_step][['close', 'open', 'high', 'low']]

        # Pass action directly to broker - it handles the logic
        # Actions: 0=HOLD, 1=LONG, 2=SHORT
        self.broker.step(
            step_index=self.current_step,
            signal=action,
            new_state=next_candle,
            tp_price=tp_price,
            sl_price=sl_price
        )

        # Calculate reward based on state transition
        reward = self.calculate_reward(action, broker_previous_state)

        done = False
        truncated = False
        if self.broker.is_bankrupt:
            print("Account bankrupt!")
            done = True
            reward = -1.0

        truncated = self.current_step >= len(self.data) - 1 if truncated is False else truncated

        self.history.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            **self.broker.get_state()
        })

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
        7. VP Bins (n_bins): Volume distribution histogram → CNN
        8. VP Levels (9): 5 continuous + 4 binary features → Split MLP
           - Continuous: high/vah/poc/val/low distances
           - Binary: in_va, above_poc, session_intersection, poc_crossover
        """
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step

        # Group 1-4: Multi-scale price patterns (tensor slicing + zero-copy conversion)
        # Use contiguous() to ensure efficient memory layout after slicing
        micro_temporal = self.micro_temporal_data[start_idx:end_idx]
        micro_spatial = self.micro_spatial_data[start_idx:end_idx]
        meso_patterns = self.meso_data[start_idx:end_idx]
        macro_patterns = self.macro_data[start_idx:end_idx]

        # Group 5: Account State
        account_state = get_account_state_features(
            self.broker.step_history,
            self.initial_balance,
            self.lookback_window
        )

        # Group 6: Position Info
        position_info = get_position_info_features(
            self.broker.step_history,
            self.lookback_window,
        )

        # Group 7 & 8: VP Bins and Levels (with shared cache across all envs)
        # Check if already calculated for this step
        if self.current_step not in self.vp_cache:
            # Calculate once and cache for all environments
            vp_bins = get_vp_bins_features_visible(
                self.data,
                self.current_step,
                self.lookback_window,
                self.n_bins
            )

            vp_levels = get_vp_levels_features_visible(
                self.data,
                self.current_step,
                self.lookback_window,
                self.n_bins
            )

            # Store in shared cache
            self.vp_cache[self.current_step] = {
                'bins': vp_bins,
                'levels': vp_levels
            }
        else:
            # Reuse from cache (other env already calculated this step)
            vp_bins = self.vp_cache[self.current_step]['bins']
            vp_levels = self.vp_cache[self.current_step]['levels']

        # Build observation dictionary (will be converted to numpy arrays below)
        obs_dict: dict = {  # type: ignore - will contain numpy arrays after conversion
            'micro_temporal': micro_temporal,
            'micro_spatial': micro_spatial,
            'meso_patterns': meso_patterns,
            'macro_patterns': macro_patterns,
            'account_state': account_state,
            'position_info': position_info,
            'vp_bins': vp_bins,
            'vp_levels': vp_levels,
        }

        """ for key, array in obs_dict.items():
            if np.isnan(array).any() or np.isinf(array).any():
                raise ValueError(f"NaN/Inf detected in observation array '{key}'.")

            # Check ranges (VP bins in [0,1], others in [-1,1])
            if key == 'vp_bins':
                expected_min, expected_max = 0.0, 1.0
            else:
                expected_min, expected_max = -1.0, 1.0

            array_min = array.min()
            array_max = array.max()
            if array_min < expected_min - 0.01 or array_max > expected_max + 0.01:
                # Find indices of invalid values using NumPy
                min_idx = np.where(array == array_min)[0].tolist()
                max_idx = np.where(array == array_max)[0].tolist()
                print(f"Warning: Values outside [{expected_min}, {expected_max}] in '{key}' at step {self.current_step}:")
                print(f"  min={array_min:.4f} at indices {min_idx[:5]}{'...' if len(min_idx) > 5 else ''}")
                print(f"  max={array_max:.4f} at indices {max_idx[:5]}{'...' if len(max_idx) > 5 else ''}") """

        return obs_dict

    def calculate_reward(self, action: int, previous_state):
        if previous_state is None:
            return 0.0

        # Check for completed trades first (highest priority)
        trade_history = self.broker.trade_history
        current_step = self.broker.current_step
        for trade in reversed(trade_history):
            if trade.get('status') == 'CLOSED' and trade.get('step_close') == current_step:
                trade_len = trade.get('step_close') - trade.get('step_open')
                self.last_open_traded_step = trade.get('step_close')
                if trade.get('reason') == 'TP':
                    # Normalize: +1.0 base reward, bonus for quick trades
                    reward = 1.0 + min(0.5, 20 / trade_len if trade_len > 0 else 0.5)
                    return np.clip(reward, -2.0, 2.0)
                elif trade.get('reason') == 'SL':
                    # Normalize: -0.5 penalty (less severe than TP reward)
                    return np.clip(-0.5, -2.0, 2.0)

        # Action-based rewards (normalized to similar scale)
        if action == 0:  # HOLD
            if previous_state.get('position_size') != 0:
                # Neutral - agent has no choice but to hold when in position
                return 0.0
            else:
                # Neutral for holding when flat
                return 0.0
        
        elif action in [1, 2]:  # LONG/SHORT
            # Small positive for taking action (exploration)
            return 0.05

        return np.clip(0.0, -2.0, 2.0)

    def calculate_reward2(self, action: int, previous_state):
        if previous_state is None:
            return 0.0

        current_state = self.broker.get_state()
        reward = 0.0
        previous_position = previous_state.get('position_size', 0)
        current_position = current_state.get('position_size', 0)
        cash_used = previous_state.get('used_balance', 1)

        # BANKRUPTCY
        if self.broker.is_bankrupt:
            return -1.0

        # Check trade history for open/close at current step
        trade_history = self.broker.trade_history
        open_reward = 0.0
        close_reward = 0.0
        current_step = self.broker.current_step

        # Find trades opened and/or closed at this step
        for trade in reversed(trade_history):
            if trade.get('status') == 'CLOSED' and trade.get('step_close') == current_step:
                pnl = trade.get('pnl', 0)
                used_balance = trade.get('position_value', cash_used)
                pnl_percentage = (pnl / used_balance) * 100 if used_balance else 0
                if pnl > 0:
                    close_reward = 0.5 * np.tanh(pnl_percentage / 5.0)
                else:
                    close_reward = -0.25 * np.tanh(abs(pnl_percentage) / 5.0)
                break  # Only reward for the most recent close
        for trade in reversed(trade_history):
            if trade.get('status') == 'OPEN' and trade.get('step_open') == current_step:
                open_reward = 0.4
                break  # Only reward for the most recent open

        # If both open and close happened, sum both rewards
        if open_reward or close_reward:
            reward = open_reward + close_reward
        # If only position opened
        elif action in [1, 2] and previous_position == 0:
            reward = 0.4
        # If only position closed
        elif close_reward:
            reward = close_reward
        # In position - unrealized PnL
        elif current_position != 0:
            unrealized_pnl = current_state.get('unrealized_pnl', 0)
            unrealized_pct = (unrealized_pnl / cash_used) * 100 if cash_used else 0
            reward = np.tanh(unrealized_pct / 10.0) * 0.2
        # Holding penalty
        else:
            reward = -0.01

        return float(np.clip(reward, -1.0, 1.0))
