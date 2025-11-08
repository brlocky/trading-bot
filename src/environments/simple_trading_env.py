"""
SimpleTradingEnv - Multi-Input Trading Environment (Phase 2: Divergence Detection + Cumulative VP)

Uses 14 feature groups with specialized CNNs for divergence detection and multi-scale volume analysis:

1. Spatial OHLC (4): Raw OHLC for Conv2d candlestick pattern detection
2. Temporal OHLC (4): Raw OHLC for Conv1d trend evolution detection
3. RSI Divergence (2): RSI, RSI_9 for divergence pattern detection via CNN
4. MACD Divergence (3): MACD, Signal, Histogram for divergence detection via CNN
5. Price Context (12): Time + candle structure + volume + spatial distances to EMAs/VWAP
6. Trend Indicators (10): EMA slopes + crossovers + price momentum
7. Momentum Oscillators (2): Stochastic K/D (RSI/MACD moved to dedicated CNNs)
8. Volume Profile (26): Summarized VP features (distances to levels, value area position, etc.)
9. Trading Sessions (3): ASIA, LONDON, NY open flags
10. Account State (4): Balance, equity, margin, total trades
11. Position Info (7): Position status, size, PnL, entry/TP/SL distances
12. Performance Metrics (7): Win rate, avg win/loss, profit factor, drawdown, sharpe, ROI
13. Daily VP Distribution (54): 50-bin rolling window volume histogram + VAH/VAL/POC/Close markers
14. Cumulative VP Distribution (54): 50-bin cumulative volume histogram + VAH/VAL/POC/Close markers

Total: 134 features per timestep + 54 daily VP bins + 54 cumulative VP bins = 242 features

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
    get_volume_profile_features,
    get_volume_profile_bins,
    precompute_price_context_features,
    precompute_trend_features,
    precompute_momentum_features,
    precompute_rsi_features,
    precompute_macd_features,
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

    def __init__(self, data, initial_balance=10000, lookback_window=288, n_bins=50, device="cuda", render_mode='human'):
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

        # Prepare data with all features - pre-compute all static features (modifies in place)
        self.price_context_cols = precompute_price_context_features(self.data)
        self.trend_feature_cols = precompute_trend_features(self.data)
        self.momentum_feature_cols = precompute_momentum_features()
        self.rsi_feature_cols = precompute_rsi_features()
        self.macd_feature_cols = precompute_macd_features(self.data)
        self.trading_session_cols = precompute_trading_sessions(self.data)

        # Action space: [direction, risk_reward_ratio, atr_multiplier]
        # - direction: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
        # - risk_reward_ratio: 0-9 (maps to 1.0x-10.0x)
        # - atr_multiplier: 0-9 (maps to 1.0-2.8)
        self.action_space = MultiDiscrete([4, 10, 10])

        self.observation_space = gym.spaces.Dict({
            'price_ohlc_spatial': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 4),
                dtype=np.float32
            ),
            'price_ohlc_temporal': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 4),
                dtype=np.float32
            ),
            'rsi_divergence': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.rsi_feature_cols)),
                dtype=np.float32
            ),
            'macd_divergence': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.macd_feature_cols)),
                dtype=np.float32
            ),
            'price_context': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.price_context_cols)),
                dtype=np.float32
            ),
            'trend_indicators': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.trend_feature_cols)),
                dtype=np.float32
            ),
            'momentum_oscillators': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, len(self.momentum_feature_cols)),
                dtype=np.float32
            ),
            'volume_profile': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 26),
                dtype=np.float32
            ),
            'vp_distribution': gym.spaces.Box(
                low=0, high=1,
                shape=(lookback_window, n_bins + 4),
                dtype=np.float32
            ),
            'trading_sessions': gym.spaces.Box(
                low=0, high=1,
                shape=(lookback_window, len(self.trading_session_cols)),
                dtype=np.float32
            ),
            'account_state': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 1),
                dtype=np.float32
            ),
            'position_info': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, 2),
                dtype=np.float32
            ),
        })

        # Initialize broker, volume profile, and zigzag state
        self.broker = SimpleBroker(initial_balance=self.initial_balance, quantity_precision=0.001)
        self.vp = EnhancedVolumeProfile(n_bins=n_bins, lookback_window=lookback_window, device=device)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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

            # Update volume profile with initial data
            self.vp.update(
                self.data['date'].iloc[i],
                self.data['open'].iloc[i],
                self.data['high'].iloc[i],
                self.data['low'].iloc[i],
                self.data['close'].iloc[i],
                self.data['volume'].iloc[i]
            )

        return self._get_obs(), {}

    def step(self, action):

        # Unpack action: [direction, risk_reward_idx, atr_idx]
        direction_action, rr_idx, atr_idx = int(action[0]), int(action[1]), int(action[2])
        risk_reward_ratio = 1.0 + (rr_idx * 1.0)  # Maps 0-9 to 1.0-10.0
        atr_multiplier = 1.0 + (atr_idx * 0.2)    # Maps 0-9 to 1.0-2.8

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

        reward = self.calculate_reward(current_state, previous_state)

        done = False
        truncated = False

        # More lenient bankruptcy check - only stop if truly bankrupt (< 5% of initial)
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

        if not truncated:
            new_date = self.data['date'].iloc[self.current_step]
            new_open = self.data['open'].iloc[self.current_step]
            new_high = self.data['high'].iloc[self.current_step]
            new_low = self.data['low'].iloc[self.current_step]
            new_close = self.data['close'].iloc[self.current_step]
            new_vol = self.data['volume'].iloc[self.current_step]

        else:
            new_date = self.data['date'].iloc[self.current_step-1]
            new_open = self.data['open'].iloc[self.current_step-1]
            new_high = self.data['high'].iloc[self.current_step-1]
            new_low = self.data['low'].iloc[self.current_step-1]
            new_close = self.data['close'].iloc[self.current_step-1]
            new_vol = self.data['volume'].iloc[self.current_step-1]

        self.vp.update(new_date, new_open, new_high, new_low, new_close, new_vol)
        obs = self._get_obs()
        return obs, reward, done, truncated, self.history[-1]

    def _get_obs(self):
        """
        Generate multi-input observation with 13 feature groups.
        Includes dual OHLC perspectives (spatial + temporal CNNs), RSI/MACD divergence CNNs, and 100-bin VP distribution.
        """
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step

        # Get data slice
        df_slice = self.data.iloc[start_idx:end_idx].copy()

        # === Group 1: Spatial OHLC (Conv2d for candlestick patterns) ===
        price_ohlc_spatial = df_slice[['open', 'high', 'low', 'close']].values
        price_ohlc_spatial = torch.tensor(price_ohlc_spatial, dtype=torch.float32, device=self.device)

        # === Group 2: Temporal OHLC (Conv1d for trend evolution) ===
        price_ohlc_temporal = df_slice[['open', 'high', 'low', 'close']].values
        price_ohlc_temporal = torch.tensor(price_ohlc_temporal, dtype=torch.float32, device=self.device)

        # === Group 3: RSI Divergence (Conv1d for RSI divergence detection) ===
        rsi_divergence = df_slice[self.rsi_feature_cols].values
        rsi_divergence = torch.tensor(rsi_divergence, dtype=torch.float32, device=self.device)

        # === Group 4: MACD Divergence (Conv1d for MACD divergence detection) ===
        macd_divergence = df_slice[self.macd_feature_cols].values
        macd_divergence = torch.tensor(macd_divergence, dtype=torch.float32, device=self.device)

        # === Group 5: Price Context (PRE-COMPUTED only, no VP here) ===
        price_context = df_slice[self.price_context_cols].values
        price_context = torch.tensor(price_context, dtype=torch.float32, device=self.device)

        # === Group 6: Trend Indicators (PRE-COMPUTED) ===
        trend_indicators = df_slice[self.trend_feature_cols].values
        trend_indicators = torch.tensor(trend_indicators, dtype=torch.float32, device=self.device)

        # === Group 7: Momentum Oscillators (PRE-COMPUTED - Stochastic only) ===
        momentum_oscillators = df_slice[self.momentum_feature_cols].values
        momentum_oscillators = torch.tensor(momentum_oscillators, dtype=torch.float32, device=self.device)

        # === Group 8: Volume Profile (all 26 VP features) ===
        current_price = float(self.data['close'].iloc[self.current_step])
        volume_profile = get_volume_profile_features(self.vp, current_price, self.lookback_window).to(self.device)

        # === Group 9: Daily VP Distribution (50 bins + 4 markers: VAH/VAL/POC/Close) ===
        close_prices = torch.tensor(df_slice['close'].values, dtype=torch.float32, device=self.device)
        vp_distribution = get_volume_profile_bins(self.vp, self.lookback_window, close_prices).to(self.device)

        # === Group 10: Trading Sessions (PRE-COMPUTED) ===
        trading_sessions = df_slice[self.trading_session_cols].values
        trading_sessions = torch.tensor(trading_sessions, dtype=torch.float32, device=self.device)

        # === Group 11: Account State ===
        account_state = get_account_state_features(
            self.broker.step_history,
            self.initial_balance,
            self.lookback_window
        ).to(self.device)

        # === Group 12: Position Info ===
        position_info = get_position_info_features(
            self.broker.step_history,
            self.lookback_window,
        ).to(self.device)

        # Build observation dictionary
        obs_dict = {
            'price_ohlc_spatial': price_ohlc_spatial,
            'price_ohlc_temporal': price_ohlc_temporal,
            'rsi_divergence': rsi_divergence,
            'macd_divergence': macd_divergence,
            'price_context': price_context,
            'trend_indicators': trend_indicators,
            'momentum_oscillators': momentum_oscillators,
            'volume_profile': volume_profile,
            'vp_distribution': vp_distribution,
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
            # Convert to CPU NumPy array for Stable-Baselines3
            obs_dict[key] = tensor.cpu().numpy().astype(np.float32)

        return obs_dict

    def calculate_reward(self, current_state, previous_state):
        """
        REWARD FUNCTION v16 - FORCE TRADING: Penalize doing nothing
        """
        if previous_state is None:
            return 0.0

        reward = 0.0
        current_trades = current_state.get('trades', [])
        previous_trades = previous_state.get('trades', [])

        # === TRADE COMPLETION REWARDS ===
        current_closed = [t for t in current_trades if t.get('status') == 'CLOSED']
        previous_closed = [t for t in previous_trades if t.get('status') == 'CLOSED']

        if len(current_closed) > len(previous_closed):
            trade = current_closed[-1]
            pnl_pct = trade.get('pnl_percent', 0)
            reason = trade.get('reason', '')

            if 'TP' in reason:
                reward = 100.0 * (1.0 + pnl_pct / 100.0)
            elif 'SL' in reason:
                reward = -20.0 * (1.0 + abs(pnl_pct) / 100.0)
            else:
                reward = -5.0

        # === HOLDING INCENTIVE ===
        current_position = current_state.get('position_size', 0)
        if current_position != 0:
            unrealized_pnl = current_state.get('unrealized_pnl', 0)
            if unrealized_pnl > 0:
                reward += 0.1
            else:
                reward -= 0.05

        # === NEW: PENALIZE INACTION (Force agent to trade) ===
        else:  # No position held
            # Count steps since last trade
            self.steps_since_close = getattr(self, 'steps_since_close', 0) + 1

            # After 100 steps of doing nothing, start penalizing
            if self.steps_since_close > 100:
                inaction_penalty = -0.01 * (self.steps_since_close - 100)  # Grows over time
                reward += max(inaction_penalty, -5.0)  # Cap at -5

        # Reset counter when trade closes
        if len(current_closed) > len(previous_closed):
            self.steps_since_close = 0

        # === BANKRUPTCY PENALTY ===
        if self.broker.is_bankrupt:
            reward = -200.0

        return float(np.clip(reward, -200.0, 200.0))

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
