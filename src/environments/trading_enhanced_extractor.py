import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TradingEnhancedExtractor(BaseFeaturesExtractor):
    """
    Rebalanced Multi-Input Feature Extractor - Optimized for Price Action Focus.

    ARCHITECTURE (6 groups):
    1. Price Patterns CNN (64-dim) - 8 features: candle structure, volume, multi-TF returns
       → 4-layer CNN (kernel=5, RF~17) + Global pooling (sees all 288 timesteps)
    2. Market Context MLP (32-dim) - 6 features: EMA/VWAP distances, volatility
    3. Trend Indicators CNN (32-dim) - 10 features: EMA slopes/crossovers (temporal momentum)
       → 4-layer CNN (kernel=5, RF~17) + Global pooling (sees all 288 timesteps)
    4. Trading Sessions Linear (3-dim) - 3 features: Session encoding
    5. Account State MLP (8-dim) - 5 features: Bounded growth/velocity features
    6. Position Info MLP (8-dim) - 7 features: Current position status

    Key Changes from v1:
    - Price Patterns: 32 → 64 dims (2x capacity for price action)
    - Market Context: 16 → 32 dims (2x capacity for trend positioning)
    - Trend Indicators: MLP → 4-layer CNN (now processes temporal momentum)
    - Trading Sessions: 4 → 3 dims (reduce over-reliance on time-of-day)
    - CNN depth: 2 → 4 layers (RF: 5 → 17 timesteps for local patterns)
    - Total embedding: 100 → 147 dims (47% increase in representation power)

    Design principles:
    - Deep CNNs for temporal patterns (price patterns, trend momentum)
      * 4 layers with kernel=5 gives ~17 timestep receptive field (local patterns)
      * AdaptiveAvgPool provides global context (all 288 timesteps)
    - MLPs for current state/positioning (distances, account, position)
    - Rebalanced to prioritize price action over simple session timing
    - ReLU activation (fast and sufficient)

    Total input: 39 features per timestep × 288 timesteps
    Total output: 147-dim embeddings → 256-dim fused features
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
        # DEEPER architecture to increase receptive field (288 timesteps)
        # With kernel=5 and 4 layers: receptive field = ~17 timesteps
        # AdaptiveAvgPool sees full 288 timesteps globally
        self.price_patterns_encoder = nn.Sequential(
            nn.Conv1d(self.shapes['price_patterns'][-1], 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling sees all 288 timesteps
        )

        # === 2. MARKET CONTEXT: Simple MLP (last timestep only) ===
        # Input: [B, T, 6] - Spatial positioning: EMA/VWAP distances, volatility
        # Current positioning matters, not how we got here
        # INCREASED from 16 to 32 dims for better market context representation
        self.market_context_encoder = nn.Sequential(
            nn.Linear(self.shapes['market_context'][-1], 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # === 3. TREND INDICATORS: 1D CNN for temporal momentum ===
        # Input: [B, T, 10] - EMA slopes (bounded via tanh) + crossovers
        # Slopes show momentum evolution over time → benefit from CNN
        # Changed from MLP to CNN to capture trend acceleration/deceleration patterns
        # DEEPER architecture for better temporal context (288 timesteps)
        # With kernel=5 and 4 layers: receptive field = ~17 timesteps
        self.trend_encoder = nn.Sequential(
            nn.Conv1d(self.shapes['trend_indicators'][-1], 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling sees all 288 timesteps
        )

        # === 4. TRADING SESSIONS: Direct embedding (last timestep) ===
        # Input: [B, T, 3] - Binary one-hot flags (Asia/London/NY)
        # REDUCED from 4 to 3 dims - too simple to deserve more capacity
        self.session_encoder = nn.Linear(self.shapes['trading_sessions'][-1], 3)

        # === 5. ACCOUNT STATE: Simple MLP (last timestep only) ===
        # Input: [B, T, 5] - Balance, equity, unrealized_pnl, realized_pnl, commission
        self.account_encoder = nn.Sequential(
            nn.Linear(self.shapes['account_state'][-1], 16),
            nn.GELU(),
            nn.Linear(16, 8),
        )

        # === 6. POSITION INFO: Simple MLP (last timestep only) ===
        # Input: [B, T, 7] - Status, leverage, pnl%, distances, risk_reward, duration
        self.position_encoder = nn.Sequential(
            nn.Linear(self.shapes['position_info'][-1], 16),
            nn.GELU(),
            nn.Linear(16, 8),
        )

        # === FUSION LAYER WITH CROSS-ATTENTION ===
        # Combined: price(64) + market(32) + trend(32) + session(3) + account(8) + position(8) = 147
        # Rebalanced: More weight on price patterns and market context
        combined_dim = 64 + 32 + 32 + 3 + 8 + 8

        # Cross-attention to learn feature relationships
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=7,  # 147 dims / 7 heads = 21 dims per head
            dropout=0.1,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(combined_dim)

        # Final fusion with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
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

        # === 3. TREND INDICATORS (CNN) ===
        # [B, T, 10] -> [B, 10, T] for Conv1d -> [B, 32, 1] -> [B, 32]
        trend_seq = obs_tensors['trend_indicators']  # [B, T, 10]
        trend_seq = trend_seq.transpose(1, 2)  # [B, 10, T]
        trend_pooled = self.trend_encoder(trend_seq).squeeze(-1)  # [B, 32]

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

        # === CROSS-ATTENTION: Learn feature relationships ===
        # Add sequence dimension for attention
        combined_seq = combined.unsqueeze(1)  # [B, 1, 147]

        # Self-attention across features (learns which features relate to each other)
        attended, _ = self.cross_attention(
            combined_seq, combined_seq, combined_seq
        )  # [B, 1, 147]

        # Apply layer norm and residual connection
        attended = attended.squeeze(1)  # [B, 147]
        combined = self.attention_norm(attended + combined)  # Residual connection

        # === FINAL FUSION ===
        fused = self.fusion(combined)  # [B, 256]

        return fused
