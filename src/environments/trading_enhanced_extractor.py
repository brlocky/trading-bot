import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TradingEnhancedExtractor(BaseFeaturesExtractor):
    """
    Rebalanced Multi-Input Feature Extractor - Optimized for Price Action Focus.

    ARCHITECTURE (6 groups):
    1. Price Patterns CNN (64-dim) - 8 features: candle structure, volume, multi-TF returns [INCREASED]
    2. Market Context MLP (32-dim) - 6 features: EMA/VWAP distances, volatility [INCREASED]
    3. Trend Indicators MLP (32-dim) - 10 features: Crossovers/momentum state
    4. Trading Sessions Linear (3-dim) - 3 features: Session encoding [REDUCED]
    5. Account State MLP (8-dim) - 5 features: Balance/equity metrics
    6. Position Info MLP (8-dim) - 7 features: Current position status

    Key Changes from Previous Version:
    - Price Patterns: 32 → 64 dims (2x capacity for price action)
    - Market Context: 16 → 32 dims (2x capacity for trend positioning)
    - Trading Sessions: 4 → 3 dims (reduce over-reliance on time-of-day)
    - Total embedding: 100 → 147 dims (47% increase in representation power)

    Design principles:
    - CNN for temporal price patterns (candle structure, volume, returns)
    - MLPs for current state/positioning (distances, trends, account, position)
    - Rebalanced to prioritize price action over simple session timing
    - ReLU activation (fast and sufficient)

    Total input: 39 features per timestep
    Total output: 147-dim embeddings -> 256-dim fused features
    """

    def __init__(self, observation_space: spaces.Dict, hidden_dim=128, **kwargs):
        out_dim = hidden_dim * 2  # 256-dim output
        super().__init__(observation_space, features_dim=out_dim)

        self.hidden_dim = hidden_dim
        self.shapes = {key: space.shape for key, space in observation_space.spaces.items()}

        # === 1. PRICE PATTERNS: 1D CNN for temporal patterns ===
        # Input: [B, T, 8] - Pure price action: candle structure, volume, multi-TF returns
        # These features benefit from temporal convolution (patterns evolve over time)
        # INCREASED from 32 to 64 dims to give price patterns more representation power
        self.price_patterns_encoder = nn.Sequential(
            nn.Conv1d(self.shapes['price_patterns'][-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Increased capacity
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool to [B, 64, 1]
        )

        # === 2. MARKET CONTEXT: Simple MLP (last timestep only) ===
        # Input: [B, T, 6] - Spatial positioning: EMA/VWAP distances, volatility
        # Current positioning matters, not how we got here
        # INCREASED from 16 to 32 dims for better market context representation
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.shapes['market_context'][-1], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # === 3. TREND INDICATORS: Simple MLP (last timestep only) ===
        # Input: [B, T, 10] - EMA slopes + crossovers + momentum
        # Binary crossovers (0/1) + continuous slopes
        self.trend_encoder = nn.Sequential(
            nn.Linear(self.shapes['trend_indicators'][-1], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # === 4. TRADING SESSIONS: Direct embedding (last timestep) ===
        # Input: [B, T, 3] - Binary one-hot flags (Asia/London/NY)
        # REDUCED from 4 to 3 dims - too simple to deserve more capacity
        self.session_encoder = nn.Linear(self.shapes['trading_sessions'][-1], 3)

        # === 5. ACCOUNT STATE: Simple MLP (last timestep only) ===
        # Input: [B, T, 5] - Balance, equity, unrealized_pnl, realized_pnl, commission
        self.account_encoder = nn.Sequential(
            nn.Linear(self.shapes['account_state'][-1], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # === 6. POSITION INFO: Simple MLP (last timestep only) ===
        # Input: [B, T, 7] - Status, leverage, pnl%, distances, risk_reward, duration
        self.position_encoder = nn.Sequential(
            nn.Linear(self.shapes['position_info'][-1], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # === FUSION LAYER ===
        # Combined: price(64) + market(32) + trend(32) + session(3) + account(8) + position(8) = 147
        # Rebalanced: More weight on price patterns and market context
        combined_dim = 64 + 32 + 32 + 3 + 8 + 8

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, observations):
        """Process all observations through specialized encoders"""
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

        # === 1. PRICE PATTERNS (CNN) ===
        # [B, T, 8] -> [B, 8, T] for Conv1d -> [B, 64, 1] -> [B, 64]
        price_seq = obs_tensors['price_patterns']  # [B, T, 8]
        price_seq = price_seq.transpose(1, 2)  # [B, 8, T]
        price_pooled = self.price_patterns_encoder(price_seq).squeeze(-1)  # [B, 64]

        # === 2. MARKET CONTEXT (MLP, last timestep) ===
        market_seq = obs_tensors['market_context']  # [B, T, 6]
        market_pooled = self.market_context_encoder(market_seq[:, -1, :])  # [B, 32]

        # === 3. TREND INDICATORS (MLP, last timestep) ===
        trend_seq = obs_tensors['trend_indicators']  # [B, T, 10]
        trend_pooled = self.trend_encoder(trend_seq[:, -1, :])  # [B, 32]

        # === 4. TRADING SESSIONS (Linear, last timestep) ===
        session_seq = obs_tensors['trading_sessions']  # [B, T, 3]
        session_pooled = self.session_encoder(session_seq[:, -1, :])  # [B, 3]

        # === 5. ACCOUNT STATE (MLP, last timestep) ===
        account_seq = obs_tensors['account_state']  # [B, T, 5]
        account_pooled = self.account_encoder(account_seq[:, -1, :])  # [B, 8]

        # === 6. POSITION INFO (MLP, last timestep) ===
        position_seq = obs_tensors['position_info']  # [B, T, 7]
        position_pooled = self.position_encoder(position_seq[:, -1, :])  # [B, 8]

        # === CONCATENATE ALL EMBEDDINGS ===
        combined = torch.cat([
            price_pooled,      # 64 (increased from 32)
            market_pooled,     # 32 (increased from 16)
            trend_pooled,      # 32
            session_pooled,    # 3 (reduced from 4)
            account_pooled,    # 8
            position_pooled,   # 8
        ], dim=1)  # [B, 147]

        # === FUSION ===
        fused = self.fusion(combined)  # [B, 256]

        return fused
