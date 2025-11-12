"""
Hybrid Multi-Scale CNN Trading Feature Extractor

Architecture:
1. Multi-Scale CNN: Specialized kernels for different temporal patterns
   - Micro Temporal CNN (kernels 3-5): OHLC + volume patterns
   - Micro Spatial Multi-Scale CNNs: Candlestick patterns at 3 scales
     * Local (kernels 3-5): 3-5 candle patterns (hammers, engulfing)
     * Medium (kernels 10-20): Consolidation, mini-trends
     * Global (adaptive pool): Session-wide structure, overall volatility
   - Meso CNN (kernels 10-15): Intraday trends, 1h-4h momentum
   - Macro CNN (kernels 30-50): Daily trends, 24h momentum
   
2. Volume Profile: Market structure via volume distribution
   - VP Bins: Volume distribution histogram
   - VP Levels: OHLC distances to key levels (VAH/POC/VAL)

Benefits:
- Temporal/Spatial Separation: Time-series → CNN, Structure → Multi-scale CNN
- Multi-scale: Each CNN optimized for specific frequency patterns
- Candlestick Patterns: Local + Medium + Global scales capture all pattern types
- Volume-Based Structure: VP captures support/resistance better than price-only swings
- Fast & Clean: No CPU graph building, pure tensor operations
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TradingHybridExtractor(BaseFeaturesExtractor):
    """
    Hybrid Multi-Scale CNN Architecture with Temporal/Spatial Separation

    Input: Multi-scale feature groups
      - Micro Temporal: OHLC + volume (5 features) → CNN
      - Micro Spatial: Body/wick ratios (4 features) → Multi-scale CNNs (local + medium + global)
      - Meso: 1h, 4h returns (2 features) → CNN
      - Macro: 24h returns (1 feature) → CNN
      - VP Bins: Volume distribution (50 bins) → CNN
      - VP Levels: OHLC distances + binary features (26 features) → Split MLP
      - Account State: (5 features) → MLP
      - Position Info: (7 features) → MLP

    Processing Paths:
      Temporal sequences → CNNs with appropriate kernel sizes
      Spatial structure → Multi-scale CNNs (3, 10, 20 candle patterns + global pooling)
      Trading state → MLPs process last timestep

    Output: Fused features combining all scales
      Total: 128-dim + VP(38) = 166-dim → hidden_dim
    """

    def __init__(self, observation_space: spaces.Dict, hidden_dim=64, **kwargs):
        self.hidden_dim = hidden_dim

        # Extract shapes from observation space
        shapes = {}
        for key, space in observation_space.spaces.items():
            shapes[key] = space.shape

        # Call parent with final output dimension
        super().__init__(observation_space, features_dim=hidden_dim)

        self.shapes = shapes

        # Extract feature dimensions for each scale
        micro_temporal_features = shapes['micro_temporal'][-1]  # 5 features (OHLC + volume)
        micro_spatial_features = shapes['micro_spatial'][-1]    # 4 features (body/wick ratios)
        meso_features = shapes['meso_patterns'][-1]             # 2 features (1h, 4h returns)
        macro_features = shapes['macro_patterns'][-1]           # 1 feature (24h return)
        vp_bins_features = shapes['vp_bins'][-1]                # 50 bins (volume distribution)
        # VP levels dimensions are hardcoded in the MLPs below (20 continuous + 6 binary)

        # === MULTI-SCALE CNN PATH: Different kernel sizes for different temporal scales ===

        # Micro Temporal CNN: Small kernels (3-5) for OHLC+volume patterns
        self.micro_temporal_cnn = nn.Sequential(
            nn.Conv1d(micro_temporal_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 32]
        )

        # Micro Spatial CNNs: Multi-scale candlestick pattern detection
        # Local patterns (3-5 candles): Immediate reversals, engulfing, hammers
        self.micro_spatial_cnn_local = nn.Sequential(
            nn.Conv1d(micro_spatial_features, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 12]
        )

        # Medium patterns (10-20 candles): Consolidation, mini-trends
        self.micro_spatial_cnn_medium = nn.Sequential(
            nn.Conv1d(micro_spatial_features, 12, kernel_size=10, padding=5),
            nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=20, padding=10),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 12]
        )

        # Global pattern (full lookback): Session-wide structure, overall volatility
        self.micro_spatial_cnn_global = nn.Sequential(
            nn.Conv1d(micro_spatial_features, 8, kernel_size=1),  # Per-candle transform
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # Best feature across entire sequence
            # → [B, 8]
        )
        # Total spatial features: 12 + 12 + 8 = 32

        # Meso CNN: Medium kernels (10-15) for intraday trends
        self.meso_cnn = nn.Sequential(
            nn.Conv1d(meso_features, 16, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 16]
        )

        # Macro CNN: Large kernels (30-50) for daily trends
        self.macro_cnn = nn.Sequential(
            nn.Conv1d(macro_features, 8, kernel_size=31, padding=15),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=51, padding=25),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 16]
        )

        # === VP PATH: Volume Profile features ===
        # VP Bins CNN: Process volume distribution evolution
        self.vp_bins_cnn = nn.Sequential(
            nn.Conv1d(vp_bins_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → [B, 16]
        )

        # VP Levels MLPs: Split continuous vs binary features
        # Last timestep only: [B, 26] → [B, 20] continuous + [B, 6] binary
        self.vp_levels_continuous_mlp = nn.Sequential(
            nn.Linear(20, 32),  # OHLC distances (20 features)
            nn.ReLU(),
            nn.Linear(32, 16)  # → [B, 16]
        )

        self.vp_levels_binary_mlp = nn.Sequential(
            nn.Linear(6, 12),  # Binary spatial features (6 features)
            nn.ReLU(),
            nn.Linear(12, 6)  # → [B, 6]
        )

        # === MLP PATHS: Account + Position ===
        self.account_encoder = nn.Sequential(
            nn.Linear(self.shapes['account_state'][-1], 16),
            nn.ReLU(),
        )

        self.position_encoder = nn.Sequential(
            nn.Linear(self.shapes['position_info'][-1], 16),
            nn.ReLU(),
        )

        # === FUSION ===
        # Combine: MicroTemporal(32) + MicroSpatial(32) + Meso(16) + Macro(16) + Account(16) + Position(16) = 128
        # + VP Bins(16) + VP Continuous(16) + VP Binary(6) = 166
        combined_dim = 32 + 32 + 16 + 16 + 16 + 16 + 16 + 16 + 6  # Multi-scale CNN + Spatial + Account + Position + VP

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        """Process observations through hybrid multi-scale CNN"""
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

        # === MULTI-SCALE CNN PATH ===
        # Micro temporal: [B, T, 5] → [B, 5, T] → CNN
        micro_temporal_seq = obs_tensors['micro_temporal'].transpose(1, 2)
        micro_temporal_features = self.micro_temporal_cnn(micro_temporal_seq).squeeze(-1)  # [B, 32]

        # Micro spatial: [B, T, 4] → [B, 4, T] → Multi-scale CNNs
        micro_spatial_seq = obs_tensors['micro_spatial'].transpose(1, 2)

        # Process through different scales
        spatial_local = self.micro_spatial_cnn_local(micro_spatial_seq).squeeze(-1)      # [B, 12]
        spatial_medium = self.micro_spatial_cnn_medium(micro_spatial_seq).squeeze(-1)    # [B, 12]
        spatial_global = self.micro_spatial_cnn_global(micro_spatial_seq).squeeze(-1)    # [B, 8]

        # Concatenate all spatial scales: 12 + 12 + 8 = 32
        micro_spatial_features = torch.cat([spatial_local, spatial_medium, spatial_global], dim=1)  # [B, 32]

        # Meso patterns: [B, T, 2] → [B, 2, T]
        meso_seq = obs_tensors['meso_patterns'].transpose(1, 2)
        meso_features = self.meso_cnn(meso_seq).squeeze(-1)  # [B, 16]

        # Macro patterns: [B, T, 1] → [B, 1, T]
        macro_seq = obs_tensors['macro_patterns'].transpose(1, 2)
        macro_features = self.macro_cnn(macro_seq).squeeze(-1)  # [B, 16]

        # === VP PATH ===
        # VP Bins: [B, T, 50] → [B, 50, T]
        vp_bins_seq = obs_tensors['vp_bins'].transpose(1, 2)
        vp_bins_features = self.vp_bins_cnn(vp_bins_seq).squeeze(-1)  # [B, 16]

        # VP Levels (last timestep): [B, T, 26] → [B, 26]
        vp_levels_last = obs_tensors['vp_levels'][:, -1, :]
        vp_levels_continuous = vp_levels_last[:, :20]  # OHLC distances (20 features)
        vp_levels_binary = vp_levels_last[:, 20:]      # Binary spatial (6 features)

        vp_continuous_features = self.vp_levels_continuous_mlp(vp_levels_continuous)  # [B, 16]
        vp_binary_features = self.vp_levels_binary_mlp(vp_levels_binary)              # [B, 6]

        # === MLP PATHS ===
        account_seq = obs_tensors['account_state']
        account_features = self.account_encoder(account_seq[:, -1, :])  # [B, 16]

        position_seq = obs_tensors['position_info']
        position_features = self.position_encoder(position_seq[:, -1, :])  # [B, 16]

        # === FUSION ===
        combined = torch.cat([
            micro_temporal_features, micro_spatial_features,  # Micro-scale (temporal + spatial)
            meso_features, macro_features,                    # Meso + Macro scale
            account_features, position_features,              # Trading state
            vp_bins_features, vp_continuous_features, vp_binary_features  # Volume Profile
        ], dim=1)

        fused = self.fusion(combined)  # [B, hidden_dim]

        return fused
