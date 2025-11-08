"""
Multi-Input Feature Extractor for Trading Environment (Phase 2: Divergence Detection + Cumulative VP + Price Context)

Processes 14 feature groups with specialized encoders:
1. OHLC Spatial CNN (4 features): Conv2d - Candlestick patterns
2. OHLC Temporal CNN (4 features): Conv1d - Price trends, momentum evolution
3. RSI Divergence CNN (2 features): Conv1d - RSI divergence pattern detection
4. MACD Divergence CNN (3 features): Conv1d - MACD divergence pattern detection
5. Price Context (12 features): Small Transformer - Time, candle structure, spatial distances
6. Trend Indicators (10 features): Transformer - EMA slopes/crossovers, price momentum
7. Momentum Oscillators (2 features): MLP - Stochastic K/D only (RSI/MACD moved to CNNs)
8. Volume Profile (26 features): Transformer - Session VP levels, naked POCs
9. Trading Sessions (3 features): MLP - ASIA, LONDON, NY flags
10. Account State (4 features): MLP - Balance, margin, equity
11. Position Info (7 features): MLP - Position details
12. Performance Metrics (7 features): MLP - Win rate, PnL, Sharpe
13. Daily VP Distribution (54 features): CNN - 50-bin volume histogram + VAH/VAL/POC/Close markers (rolling window)
14. Cumulative VP Distribution (54 features): CNN - 50-bin cumulative volume + VAH/VAL/POC/Close markers (all-time)

Total: 134 features per timestep across 14 groups (242 total including VP bins with price context)
Lookback: 288 timesteps

INDEPENDENT CNN Architecture for Oscillators + VP:
- Group 1 - Spatial (Conv2d): Candlestick patterns → 32-dim
- Group 2 - Temporal (Conv1d): Trend evolution → 64-dim
- Group 3 - RSI Divergence (Conv1d): RSI divergence detection → 32-dim
- Group 4 - MACD Divergence (Conv1d): MACD divergence detection → 32-dim
- Group 13 - Daily VP Bins (Conv1d): Intraday volume patterns + price position → 16-dim
- Group 14 - Cumulative VP Bins (Conv1d): All-time accumulation zones + price position → 16-dim

Output: 332-dim combined embedding → 256-dim fused features
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialOHLCCNN(nn.Module):
    """
    SPATIAL CNN (Conv2d) for OHLC candlestick pattern detection.

    Treats OHLC as a 2D image to learn cross-channel patterns at each timestep:
    - Candlestick patterns (doji, hammer, engulfing, shooting star)
    - OHLC relationships: "High-Low spread at Open level"
    - Multi-candle formations: Morning star, evening star, three white soldiers
    - Volume-price divergences

    Input: [batch, seq_len, 4] (OHLC) - seq_len=288 (24 hours)
    Output: [batch, 32] (spatial pattern features)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.spatial_cnn = nn.Sequential(
            # Input: (batch, 1, lookback, 4) - treat OHLC as 2D image
            # Learn patterns across all 4 channels at each timestep
            nn.Conv2d(1, 16, kernel_size=(3, 4), padding=(1, 0)),  # 3 timesteps, all OHLC
            nn.GroupNorm(4, 16),  # GroupNorm instead of BatchNorm - not affected by batch correlation
            nn.GELU(),

            nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(2, 0)),  # 5 timesteps, per-channel refinement
            nn.GroupNorm(8, 32),  # GroupNorm instead of BatchNorm
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7 timesteps, deeper patterns
            nn.GroupNorm(8, 32),  # GroupNorm instead of BatchNorm
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # Pool to single value per channel
            nn.Flatten(),  # (batch, 32)
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 4] OHLC tensor (seq_len=288)

        Returns:
            embedding: [batch, 32] spatial features
        """
        # Add channel dimension: [B, T, 4] → [B, 1, T, 4]
        x = x.unsqueeze(1)
        x = self.spatial_cnn(x)  # [B, 32]
        x = self.output_proj(x)  # [B, hidden_dim]
        return x


class TemporalOHLCCNN(nn.Module):
    """
    TEMPORAL CNN (Conv1d) for OHLC evolution over time.

    Each OHLC channel evolves independently to learn:
    - Price trends: "Close rising for 10 candles"
    - Volatility patterns: "High-Low range expanding"
    - Support/resistance: "Lows bouncing at same level"
    - Momentum shifts: "Open-Close delta accelerating"

    Uses multi-scale parallel paths + dilated convolutions for long-range dependencies.

    Input: [batch, seq_len, 4] (OHLC) - seq_len=288 (24 hours)
    Output: [batch, 64] (temporal pattern features)
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Multi-scale parallel paths (different kernel sizes)
        self.temporal_paths = nn.ModuleList([
            # 1x1: Instantaneous OHLC relationships
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=1, padding=0),
                nn.GroupNorm(3, 12),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # 3x3: Very short patterns (15min)
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=3, padding=1),
                nn.GroupNorm(3, 12),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # 5x5: Short patterns (25min)
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=5, padding=2),
                nn.GroupNorm(3, 12),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # 7x7: Medium patterns (35min)
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=7, padding=3),
                nn.GroupNorm(3, 12),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # 11x11: Long patterns (55min)
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=11, padding=5),
                nn.GroupNorm(3, 12),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # 15x15: Very long patterns (1.25hr)
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=15, padding=7),
                nn.GroupNorm(3, 12),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
        ])

        # Dilated temporal convolutions for MACRO patterns (hours-long trends)
        # 12*6 = 72 channels from parallel paths
        self.temporal_dilated = nn.Sequential(
            # Dilation=2: sees every 2nd timestep (10min apart)
            nn.Conv1d(72, 64, kernel_size=3, dilation=2, padding=2),
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm - critical fix!
            nn.GELU(),

            # Dilation=4: sees every 4th timestep (20min apart)
            nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm
            nn.GELU(),

            # Dilation=8: sees every 8th timestep (40min apart)
            nn.Conv1d(64, 64, kernel_size=3, dilation=8, padding=8),
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm
            nn.GELU(),

            # Global pooling over time
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # (batch, 64)
        )

        self.output_proj = nn.Linear(64, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 4] OHLC tensor (seq_len=288)

        Returns:
            embedding: [batch, 64] temporal features
        """
        # Transpose for Conv1d: [B, T, 4] → [B, 4, T]
        x = x.permute(0, 2, 1)

        # Apply parallel temporal convolutions
        temporal_outputs = []
        for path in self.temporal_paths:
            temp_out = path(x)  # [B, 12, T]
            temporal_outputs.append(temp_out)

        # Concatenate parallel outputs: [B, 72, T]
        x = torch.cat(temporal_outputs, dim=1)

        # Apply dilated convolutions for long-range dependencies
        x = self.temporal_dilated(x)  # [B, 64]

        # Final projection
        x = self.output_proj(x)  # [B, hidden_dim]

        return x


class RSI_DivergenceCNN(nn.Module):
    """
    RSI DIVERGENCE CNN for detecting bullish/bearish divergences.

    Learns patterns like:
    - Regular Bullish Divergence: Price makes lower low, RSI makes higher low
    - Regular Bearish Divergence: Price makes higher high, RSI makes lower high
    - Hidden Bullish Divergence: Price makes higher low, RSI makes lower low
    - Hidden Bearish Divergence: Price makes lower high, RSI makes higher high

    Input: [batch, seq_len, 1] (RSI only) - seq_len=288 (24 hours)
    Output: [batch, 32] (divergence features)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Multi-scale parallel paths for divergence detection
        self.divergence_paths = nn.ModuleList([
            # Short-term divergences: 5-bar lookback (~25 minutes)
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, padding=2),
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # Medium-term divergences: 11-bar lookback (~55 minutes) - ODD kernel for exact padding
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=11, padding=5),  # (11-1)//2 = 5
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # Long-term divergences: 21-bar lookback (~105 minutes) - ODD kernel
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=21, padding=10),  # (21-1)//2 = 10
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # Very long-term divergences: 41-bar lookback (~205 minutes) - ODD kernel
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=41, padding=20),  # (41-1)//2 = 20
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
        ])

        # Temporal aggregation with dilated convolutions
        # 8*4 = 32 channels from parallel paths
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.GroupNorm(4, 32),  # GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(4, 32),  # GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # [B, 32]
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 2] RSI tensor (RSI, RSI_9)

        Returns:
            embedding: [batch, 32] divergence features
        """
        # Transpose for Conv1d: [B, T, 2] → [B, 2, T]
        x = x.permute(0, 2, 1)

        # Apply parallel divergence detection paths
        divergence_outputs = []
        for path in self.divergence_paths:
            divergence_outputs.append(path(x))  # Each: [B, 8, T]

        # Concatenate parallel outputs: [B, 32, T]
        x = torch.cat(divergence_outputs, dim=1)

        # Temporal fusion
        x = self.temporal_fusion(x)  # [B, 32]

        # Final projection
        x = self.output_proj(x)  # [B, hidden_dim]

        return x


class MACD_DivergenceCNN(nn.Module):
    """
    MACD DIVERGENCE CNN for detecting MACD-based divergences and momentum shifts.

    Learns patterns like:
    - MACD divergences (MACD vs price)
    - Histogram divergences (hidden divergences)
    - MACD crossovers (MACD crossing signal line)
    - Histogram momentum shifts (acceleration/deceleration)

    Input: [batch, seq_len, 3] (MACD, MACD_Signal, MACD_Histogram) - seq_len=288
    Output: [batch, 32] (divergence features)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Multi-scale parallel paths for MACD divergence detection
        self.divergence_paths = nn.ModuleList([
            # Short-term: 5-bar patterns (~25 minutes)
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=5, padding=2),
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # Medium-term: 11-bar patterns (~55 minutes) - ODD kernel
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=11, padding=5),  # (11-1)//2 = 5
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # Long-term: 21-bar patterns (~105 minutes) - ODD kernel
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=21, padding=10),  # (21-1)//2 = 10
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
            # Very long-term: 41-bar patterns (~205 minutes) - ODD kernel
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=41, padding=20),  # (41-1)//2 = 20
                nn.GroupNorm(2, 8),  # GroupNorm instead of BatchNorm
                nn.GELU(),
            ),
        ])

        # Temporal aggregation with dilated convolutions
        # 8*4 = 32 channels from parallel paths
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.GroupNorm(4, 32),  # GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(4, 32),  # GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # [B, 32]
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 3] MACD tensor (MACD, Signal, Histogram)

        Returns:
            embedding: [batch, 32] divergence features
        """
        # Transpose for Conv1d: [B, T, 3] → [B, 3, T]
        x = x.permute(0, 2, 1)

        # Apply parallel divergence detection paths
        divergence_outputs = []
        for path in self.divergence_paths:
            divergence_outputs.append(path(x))  # Each: [B, 8, T]

        # Concatenate parallel outputs: [B, 32, T]
        x = torch.cat(divergence_outputs, dim=1)

        # Temporal fusion
        x = self.temporal_fusion(x)  # [B, 32]

        # Final projection
        x = self.output_proj(x)  # [B, hidden_dim]

        return x


class TradingCombinedExtractor(BaseFeaturesExtractor):
    """
    Multi-input feature extractor with specialized encoders per feature group.

    Architecture:
        Each feature group → Specialized Encoder → Embedding
        All embeddings → Concatenate → Fusion Layer → Output

    12 groups: price_ohlc_spatial, price_ohlc_temporal, rsi_divergence, macd_divergence,
               price_context, trend_indicators, momentum_oscillators, volume_profile,
               trading_sessions, account_state, position_info, vp_distribution
    """

    def __init__(self, observation_space: spaces.Dict, hidden_dim=128, **kwargs):
        # Calculate total output dimension
        out_dim = hidden_dim * 2  # Will be concatenated and then projected
        super().__init__(observation_space, features_dim=out_dim)

        self.hidden_dim = hidden_dim

        # Get shapes for each group
        self.shapes = {key: space.shape for key, space in observation_space.spaces.items()}

        # === Group-Specific Encoders ===

        # 1. OHLC Spatial CNN (Conv2d): Candlestick patterns (4 features)
        # Output: 32-dim spatial features
        self.ohlc_spatial_cnn = SpatialOHLCCNN(hidden_dim=32)

        # 2. OHLC Temporal CNN (Conv1d): Trend evolution (4 features)
        # Output: 64-dim temporal features
        self.ohlc_temporal_cnn = TemporalOHLCCNN(hidden_dim=64)

        # 3. RSI Divergence CNN: Divergence pattern detection (2 features: RSI, RSI_9)
        # Output: 32-dim divergence features
        self.rsi_divergence_cnn = RSI_DivergenceCNN(hidden_dim=32)

        # 4. MACD Divergence CNN: MACD divergence detection (3 features: MACD, Signal, Hist)
        # Output: 32-dim divergence features
        self.macd_divergence_cnn = MACD_DivergenceCNN(hidden_dim=32)

        # 5. Price Context: Transformer for temporal patterns (12 features)
        self.price_projection = nn.Linear(self.shapes['price_context'][-1], 64)
        price_encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.price_transformer = nn.TransformerEncoder(
            price_encoder_layer,
            num_layers=2,
            enable_nested_tensor=False  # Disable to avoid warning with norm_first=True
        )
        self.price_output = nn.Linear(64, 32)

        # 6. Trend Indicators: Transformer for momentum patterns (10 features)
        self.trend_projection = nn.Linear(self.shapes['trend_indicators'][-1], 64)
        trend_encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.trend_transformer = nn.TransformerEncoder(
            trend_encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )
        self.trend_output = nn.Linear(64, 32)

        # 7. Momentum Oscillators: MLP (2 features - Stochastic K/D only)
        self.momentum_encoder = nn.Sequential(
            nn.Linear(self.shapes['momentum_oscillators'][-1], 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.GELU(),
        )

        # 8. Volume Profile: Transformer for distribution understanding (26 features)
        # Features: Current/prev day VAH/VAL/POC, value area position, naked POCs/VAH/VAL
        self.vp_projection = nn.Linear(self.shapes['volume_profile'][-1], 48)
        vp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=48,
            nhead=4,
            dim_feedforward=96,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.vp_transformer = nn.TransformerEncoder(
            vp_encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
        )
        self.vp_output = nn.Linear(48, 24)

        # 9. Trading Sessions: Simple MLP for session flags (3 features)
        self.session_encoder = nn.Sequential(
            nn.Linear(self.shapes['trading_sessions'][-1], 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
        )

        # 10. Account State: Simple MLP (1 feature - equity only)
        self.account_encoder = nn.Sequential(
            nn.Linear(self.shapes['account_state'][-1], 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
        )

        # 11. Position Info: MLP (2 features - status + size only)
        self.position_encoder = nn.Sequential(
            nn.Linear(self.shapes['position_info'][-1], 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
        )

        # 12. Volume Profile Bins: 1D CNN for spatial patterns (50 bins + 4 markers = 54 features)
        # Full temporal resolution (288 timesteps) - no downsampling
        self.vp_bins_cnn = nn.Sequential(
            nn.Conv1d(54, 64, kernel_size=5, padding=2),  # [B, 54, 288] → [B, 64, 288]
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),  # [B, 64, 288] → [B, 64, 288]
            nn.GroupNorm(8, 64),  # GroupNorm instead of BatchNorm
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),  # [B, 64, 288] → [B, 32, 288]
            nn.GroupNorm(4, 32),  # GroupNorm instead of BatchNorm
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # [B, 32, 288] → [B, 32, 1]
        )
        self.vp_bins_output = nn.Linear(32, 16)

        # 13. Cumulative Volume Profile Bins: REMOVED for performance (was taking 78% of _get_obs() time)
        # self.cumulative_vp_bins_downsample = nn.Conv1d(54, 54, kernel_size=9, stride=9, groups=54)
        # self.cumulative_vp_bins_cnn = nn.Sequential(...)
        # self.cumulative_vp_bins_output = nn.Linear(32, 16)

        # === Temporal Pooling Layers ===
        # Use separate attention modules for different feature dimensions to avoid truncation
        self.temporal_attention_24 = nn.MultiheadAttention(
            embed_dim=24,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_attention_32 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Learnable queries for attention pooling (per dimension)
        self.pool_query_24 = nn.Parameter(torch.randn(1, 1, 24) * 0.01)
        self.pool_query_32 = nn.Parameter(torch.randn(1, 1, 32) * 0.01)

        # === Fusion Layer ===
        # Total: 32(OHLC_Spatial) + 64(OHLC_Temporal) + 32(RSI_Div) + 32(MACD_Div) + 32(price) + 32(trend) + 24(momentum) + 24(VP) +
        #        4(sessions) + 4(account) + 4(position) + 16(VP_bins) = 300
        combined_dim = 32 + 64 + 32 + 32 + 32 + 32 + 24 + 24 + 4 + 4 + 4 + 16

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
        )

    def forward(self, observations):
        """
        Process multi-input observations through specialized encoders.

        Args:
            observations: Dict with keys matching observation space
                Each value: [batch, seq_len, features]

        Returns:
            features: [batch, hidden_dim*2]
        """
        # Get the device from model parameters
        device = next(self.parameters()).device

        # Convert numpy arrays to tensors and move to correct device
        obs_tensors = {}
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.from_numpy(value).float().to(device)
            elif isinstance(value, torch.Tensor):
                obs_tensors[key] = value.to(device)
            else:
                obs_tensors[key] = value

        # === Process Each Feature Group ===

        # 1. OHLC Spatial CNN: [B, T, 4] → Conv2d → [B, 32]
        ohlc_seq = obs_tensors['price_ohlc_spatial']
        ohlc_spatial_encoded = self.ohlc_spatial_cnn(ohlc_seq)  # [B, 32]

        # 2. OHLC Temporal CNN: [B, T, 4] → Conv1d → [B, 64]
        ohlc_temporal_seq = obs_tensors['price_ohlc_temporal']
        ohlc_temporal_encoded = self.ohlc_temporal_cnn(ohlc_temporal_seq)  # [B, 64]

        # 3. RSI Divergence CNN: [B, T, 2] → Conv1d → [B, 32]
        rsi_seq = obs_tensors['rsi_divergence']
        rsi_divergence_encoded = self.rsi_divergence_cnn(rsi_seq)  # [B, 32]

        # 4. MACD Divergence CNN: [B, T, 3] → Conv1d → [B, 32]
        macd_seq = obs_tensors['macd_divergence']
        macd_divergence_encoded = self.macd_divergence_cnn(macd_seq)  # [B, 32]

        # 5. Price Context: [B, T, 12] → Transformer → [B, 32]
        price_seq = obs_tensors['price_context']
        price_proj = self.price_projection(price_seq)  # [B, T, 64]
        price_transformed = self.price_transformer(price_proj)  # [B, T, 64]
        price_encoded = self.price_output(price_transformed)  # [B, T, 32]
        price_pooled = self._pool_temporal(price_encoded, method='attention')

        # 4. Trend Indicators: [B, T, 10] → Transformer → [B, 32]
        trend_seq = obs_tensors['trend_indicators']
        trend_proj = self.trend_projection(trend_seq)  # [B, T, 64]
        trend_transformed = self.trend_transformer(trend_proj)  # [B, T, 64]
        trend_encoded = self.trend_output(trend_transformed)  # [B, T, 32]
        trend_pooled = self._pool_temporal(trend_encoded, method='attention')

        # 4. Momentum Oscillators: [B, T, 7] → [B, 24]
        momentum_seq = obs_tensors['momentum_oscillators']
        momentum_encoded = self.momentum_encoder(momentum_seq)  # [B, T, 24]
        momentum_pooled = self._pool_temporal(momentum_encoded, method='last')

        # 5. Volume Profile: [B, T, 26] → Transformer → [B, 24]
        vp_seq = obs_tensors['volume_profile']
        vp_proj = self.vp_projection(vp_seq)  # [B, T, 48]
        vp_transformed = self.vp_transformer(vp_proj)  # [B, T, 48]
        vp_encoded = self.vp_output(vp_transformed)  # [B, T, 24]
        vp_pooled = self._pool_temporal(vp_encoded, method='attention')

        # 6. Trading Sessions: [B, T, 3] → [B, 4]
        session_seq = obs_tensors['trading_sessions']
        session_encoded = self.session_encoder(session_seq)  # [B, T, 4]
        session_pooled = session_encoded[:, -1, :]  # Just take last timestep

        # 7. Account State: [B, T, 1] → [B, 4]
        account_seq = obs_tensors['account_state']
        account_encoded = self.account_encoder(account_seq)  # [B, T, 4]
        account_pooled = account_encoded[:, -1, :]  # Just take last timestep

        # 8. Position Info: [B, T, 2] → [B, 4]
        position_seq = obs_tensors['position_info']
        position_encoded = self.position_encoder(position_seq)  # [B, T, 4]
        position_pooled = position_encoded[:, -1, :]  # Just take last timestep

        # 9. Volume Profile Bins: [B, T, 54] → CNN → [B, 16]
        vp_bins_seq = obs_tensors['vp_distribution']  # [B, T, 54] - 50 bins + 4 markers
        vp_bins_transposed = vp_bins_seq.transpose(1, 2)  # [B, 54, 288]
        vp_bins_cnn_out = self.vp_bins_cnn(vp_bins_transposed)  # [B, 32, 1]
        vp_bins_encoded = vp_bins_cnn_out.squeeze(-1)  # [B, 32]
        vp_bins_pooled = self.vp_bins_output(vp_bins_encoded)  # [B, 16]

        # 11. Cumulative Volume Profile Bins: REMOVED for performance
        # cumulative_vp_bins_seq = obs_tensors['cumulative_vp_distribution']
        # cumulative_vp_bins_transposed = cumulative_vp_bins_seq.transpose(1, 2)
        # cumulative_vp_bins_downsampled = self.cumulative_vp_bins_downsample(cumulative_vp_bins_transposed)
        # cumulative_vp_bins_cnn_out = self.cumulative_vp_bins_cnn(cumulative_vp_bins_downsampled)
        # cumulative_vp_bins_encoded = cumulative_vp_bins_cnn_out.squeeze(-1)
        # cumulative_vp_bins_pooled = self.cumulative_vp_bins_output(cumulative_vp_bins_encoded)

        # === Concatenate All Embeddings ===
        combined = torch.cat([
            ohlc_spatial_encoded,      # 32
            ohlc_temporal_encoded,     # 64
            rsi_divergence_encoded,    # 32
            macd_divergence_encoded,   # 32
            price_pooled,              # 32
            trend_pooled,              # 32
            momentum_pooled,           # 24
            vp_pooled,                 # 24
            session_pooled,            # 4
            account_pooled,            # 4
            position_pooled,           # 4
            vp_bins_pooled,            # 16
        ], dim=1)  # [B, 300] (reduced from 316)

        # === Fusion Layer ===
        fused = self.fusion(combined)  # [B, hidden_dim*2]

        return fused

    def _pool_temporal(self, x, method='last'):
        """
        Pool temporal dimension.

        Args:
            x: [batch, seq_len, features]
            method: 'last', 'mean', 'max', 'attention'

        Returns:
            pooled: [batch, features]
        """
        if method == 'last':
            return x[:, -1, :]
        elif method == 'mean':
            return x.mean(dim=1)
        elif method == 'max':
            return x.max(dim=1)[0]
        elif method == 'attention':
            batch_size = x.shape[0]
            feat_dim = x.shape[2]

            # Use appropriate attention module based on feature dimension
            if feat_dim == 24:
                attention_module = self.temporal_attention_24
                query = self.pool_query_24.expand(batch_size, -1, -1)
            elif feat_dim == 32:
                attention_module = self.temporal_attention_32
                query = self.pool_query_32.expand(batch_size, -1, -1)
            else:
                # Fallback to 'last' for unsupported dimensions
                return x[:, -1, :]

            pooled, _ = attention_module(query, x, x)
            pooled = pooled.squeeze(1)  # [B, feat_dim]

            return pooled
        else:
            raise ValueError(f"Unknown pooling method: {method}")


class CompactTradingExtractor(BaseFeaturesExtractor):
    """
    Simpler version with just MLPs for faster training.
    Use this if TradingCombinedExtractor is too slow.
    Supports 11 feature groups including dual OHLC CNNs.
    """

    def __init__(self, observation_space: spaces.Dict, hidden_dim=128, **kwargs):
        out_dim = hidden_dim * 2
        super().__init__(observation_space, features_dim=out_dim)

        self.hidden_dim = hidden_dim
        self.shapes = {key: space.shape for key, space in observation_space.spaces.items()}

        # Simple MLP encoder for each group
        self.encoders = nn.ModuleDict()
        embed_sizes = {
            'price_ohlc_spatial': 32,    # Spatial OHLC
            'price_ohlc_temporal': 64,   # Temporal OHLC
            'price_context': 32,
            'trend_indicators': 32,
            'momentum_oscillators': 24,
            'volume_profile': 24,
            'trading_sessions': 4,
            'account_state': 4,          # Changed from 8 (1 feature now)
            'position_info': 4,          # Changed from 8 (2 features now)
            'vp_distribution': 16,       # VP bins (named vp_distribution in env)
        }

        for key, embed_size in embed_sizes.items():
            input_size = self.shapes[key][1]  # feature dimension
            self.encoders[key] = nn.Sequential(
                nn.Linear(input_size, embed_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_size * 2, embed_size),
                nn.ReLU(),
            )

        # Fusion
        combined_dim = sum(embed_sizes.values())  # 240 (reduced from 252)
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
        )

    def forward(self, observations):
        """Simple forward: encode last timestep of each group, concatenate, fuse."""
        # Get the device from model parameters
        device = next(self.parameters()).device

        embeddings = []

        for key, encoder in self.encoders.items():
            obs = observations[key]
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(device)
            elif isinstance(obs, torch.Tensor):
                obs = obs.to(device)

            # Take last timestep: [B, T, F] → [B, F]
            last_step = obs[:, -1, :]

            # Encode
            embed = encoder(last_step)
            embeddings.append(embed)

        # Concatenate and fuse
        combined = torch.cat(embeddings, dim=1)
        fused = self.fusion(combined)

        return fused
