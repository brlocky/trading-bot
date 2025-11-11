"""
Minimal Trading Feature Extractor - Raw OHLCV Only

Strips out all pre-computed indicators.
Only processes raw candle data + account/position state.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TradingMinimalExtractor(BaseFeaturesExtractor):
    """
    Minimal feature extractor using only raw data:
    - Raw OHLCV candles (no indicators!)
    - Account state (balance, equity, pnl)
    - Position info (current trade status)

    Total input: 6 candle features + 5 account + 7 position = 18 features
    Much simpler than 39+ features with pre-computed indicators!
    """

    def __init__(self, observation_space: spaces.Dict, hidden_dim=64, **kwargs):
        # Output will be hidden_dim (not hidden_dim * 2)
        super().__init__(observation_space, features_dim=hidden_dim)

        self.hidden_dim = hidden_dim
        self.shapes = {key: space.shape for key, space in observation_space.spaces.items()}

        # === 1. RAW CANDLE DATA: 1D CNN ===
        # Input: [B, T, C] where C = candle features (open, high, low, close, volume, etc.)
        # Process temporal patterns in raw price action
        candle_features = self.shapes['price_patterns'][-1]

        self.candle_encoder = nn.Sequential(
            nn.Conv1d(candle_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )

        # === 2. ACCOUNT STATE: Simple MLP ===
        # Input: [B, T, 5] - balance, equity, unrealized_pnl, realized_pnl, commission
        # Use last timestep only
        self.account_encoder = nn.Sequential(
            nn.Linear(self.shapes['account_state'][-1], 16),
            nn.ReLU(),
        )

        # === 3. POSITION INFO: Simple MLP ===
        # Input: [B, T, 7] - position status, pnl, distances, etc.
        # Use last timestep only
        self.position_encoder = nn.Sequential(
            nn.Linear(self.shapes['position_info'][-1], 16),
            nn.ReLU(),
        )

        # === FUSION LAYER ===
        # Combined: candle(64) + account(16) + position(16) = 96
        combined_dim = 64 + 16 + 16

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        """Process all observations through simple encoders"""
        device = next(self.parameters()).device

        # Convert to tensors
        obs_tensors = {}
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.from_numpy(value).float().to(device)
            elif isinstance(value, torch.Tensor):
                obs_tensors[key] = value.to(device)
            else:
                obs_tensors[key] = value

        # === 1. CANDLE DATA (CNN) ===
        # [B, T, C] -> [B, C, T] for Conv1d -> [B, 64, 1] -> [B, 64]
        candle_seq = obs_tensors['price_patterns']  # [B, T, C]
        candle_seq = candle_seq.transpose(1, 2)  # [B, C, T]
        candle_features = self.candle_encoder(candle_seq).squeeze(-1)  # [B, 64]

        # === 2. ACCOUNT STATE (MLP, last timestep) ===
        account_seq = obs_tensors['account_state']  # [B, T, 5]
        account_features = self.account_encoder(account_seq[:, -1, :])  # [B, 16]

        # === 3. POSITION INFO (MLP, last timestep) ===
        position_seq = obs_tensors['position_info']  # [B, T, 7]
        position_features = self.position_encoder(position_seq[:, -1, :])  # [B, 16]

        # === CONCATENATE ALL EMBEDDINGS ===
        combined = torch.cat([
            candle_features,    # 64
            account_features,   # 16
            position_features,  # 16
        ], dim=1)  # [B, 96]

        # === FUSION ===
        fused = self.fusion(combined)  # [B, hidden_dim]

        return fused
