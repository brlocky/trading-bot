"""
Enhanced Multi-Input Feature Extractor with Pattern Recognition (Phase 3: Elliott Wave + Ranges + Reversal Patterns)

NEW PATTERN DETECTION MODULES:
1. Range Detection CNN - Consolidation zones, support/resistance
2. Elliott Wave Pattern CNN - 5-wave impulse, 3-wave correction (ABC)
3. Reversal Pattern CNN - H&S, Double Top/Bottom, Flags, Pennants
4. Support/Resistance Memory - Key price levels with interaction tracking

ARCHITECTURE IMPROVEMENTS:
- Squeeze-and-Excitation (SE) blocks for channel attention
- Residual connections for better gradient flow
- Multi-scale range detection (5m, 15m, 1h, 4h equivalent ranges)
- Wave counting mechanism for Elliott Wave patterns

Total: Previous 134 features + 3 new pattern groups (64 additional dims)
Output: 364-dim combined embedding → 256-dim fused features
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Channel Attention.

    Learns to emphasize important pattern channels and suppress noise.
    Critical for pattern recognition - helps model focus on active patterns.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),  # Min 4 channels
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T] or [B, C, H, W]
        Returns:
            Attention-weighted x: same shape as input
        """
        b, c = x.shape[:2]
        # Global average pooling
        squeeze = x.view(b, c, -1).mean(dim=2)  # [B, C]
        excitation = self.fc(squeeze).view(b, c, *([1]*(len(x.shape)-2)))
        return x * excitation


class RangeDetectionCNN(nn.Module):
    """
    RANGE DETECTION CNN - Identifies consolidation zones and breakouts.

    Detects:
    - Tight ranges (5-15 candles): Accumulation/distribution
    - Medium ranges (15-30 candles): Continuation patterns
    - Wide ranges (30-60 candles): Major support/resistance zones
    - Range position: Price at top/middle/bottom of range
    - Range strength: How many times price bounced
    - Breakout probability: Volatility compression patterns

    Input: [batch, seq_len, 4] (OHLC) - seq_len=288
    Output: [batch, 32] (range features)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Multi-scale range detection
        self.range_detectors = nn.ModuleList([
            # Tight ranges: 5 candles (~25 min)
            nn.Sequential(
                nn.Conv1d(4, 8, kernel_size=5, padding=2),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            # Short ranges: 11 candles (~55 min)
            nn.Sequential(
                nn.Conv1d(4, 8, kernel_size=11, padding=5),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            # Medium ranges: 21 candles (~105 min)
            nn.Sequential(
                nn.Conv1d(4, 8, kernel_size=21, padding=10),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            # Wide ranges: 41 candles (~205 min)
            nn.Sequential(
                nn.Conv1d(4, 8, kernel_size=41, padding=20),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
        ])

        # Volatility compression detector (squeeze before breakout)
        self.volatility_detector = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=15, padding=7),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        # SE block for pattern attention
        self.se_block = SEBlock(32, reduction=4)

        # Range position detector (where is price in the range?)
        self.position_detector = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 4] OHLC tensor
        Returns:
            embedding: [batch, hidden_dim] range features
        """
        # Transpose for Conv1d: [B, T, 4] → [B, 4, T]
        x = x.permute(0, 2, 1)

        # Detect ranges at multiple scales
        range_outputs = []
        for detector in self.range_detectors:
            range_outputs.append(detector(x))  # Each: [B, 8, T]

        # Concatenate: [B, 32, T]
        x = torch.cat(range_outputs, dim=1)

        # Detect volatility compression
        x = self.volatility_detector(x)  # [B, 32, T]

        # Apply channel attention
        x = self.se_block(x)  # [B, 32, T]

        # Detect range position
        x = self.position_detector(x)  # [B, 32]

        # Final projection
        x = self.output_proj(x)  # [B, hidden_dim]

        return x


class ElliottWaveCNN(nn.Module):
    """
    ELLIOTT WAVE PATTERN DETECTOR - Identifies impulse (12345) and correction (ABC) waves.

    Detects:
    - 5-wave IMPULSE patterns:
        * Wave 1: Initial move (15-25 candles)
        * Wave 2: Pullback ~50-61.8% (10-15 candles)
        * Wave 3: Strongest move (25-40 candles) - EXTENDED
        * Wave 4: Pullback ~38.2% (10-15 candles)
        * Wave 5: Final push (15-25 candles)

    - 3-wave CORRECTION patterns (ABC):
        * Wave A: Initial correction (15-25 candles)
        * Wave B: Counter-trend bounce (10-15 candles)
        * Wave C: Final correction (20-30 candles)

    - Wave characteristics:
        * Wave 3 must be longest (impulse)
        * Wave 2 cannot retrace 100% of Wave 1
        * Wave 4 cannot enter Wave 1 territory
        * Fibonacci ratios between waves

    Input: [batch, seq_len, 4] (OHLC) - seq_len=288
    Output: [batch, 48] (wave features: 32 impulse + 16 correction)
    """

    def __init__(self, hidden_dim=48):
        super().__init__()

        # === IMPULSE WAVE DETECTORS (12345) ===

        # Wave 1 detector: Initial move (15-25 candles)
        self.wave1_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=21, padding=10),  # ~21 candle pattern
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Wave 2 detector: Pullback pattern (10-15 candles)
        self.wave2_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=13, padding=6),  # ~13 candle pattern
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Wave 3 detector: EXTENDED move (25-40 candles) - LONGEST
        self.wave3_detector = nn.Sequential(
            nn.Conv1d(4, 12, kernel_size=33, padding=16),  # ~33 candle pattern (WIDE)
            nn.GroupNorm(3, 12),
            nn.GELU(),
        )

        # Wave 4 detector: Shallow pullback (10-15 candles)
        self.wave4_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=13, padding=6),  # ~13 candle pattern
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Wave 5 detector: Final push (15-25 candles)
        self.wave5_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=21, padding=10),  # ~21 candle pattern
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Impulse fusion: Combine 5 waves (8+8+12+8+8 = 44 channels)
        self.impulse_fusion = nn.Sequential(
            nn.Conv1d(44, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        # SE block for impulse patterns
        self.impulse_se = SEBlock(32, reduction=4)

        # === CORRECTION WAVE DETECTORS (ABC) ===

        # Wave A detector: Initial correction (15-25 candles)
        self.waveA_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=21, padding=10),
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Wave B detector: Counter-trend bounce (10-15 candles)
        self.waveB_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=13, padding=6),
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Wave C detector: Final correction (20-30 candles)
        self.waveC_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=25, padding=12),
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # Correction fusion: Combine 3 waves (8+8+8 = 24 channels)
        self.correction_fusion = nn.Sequential(
            nn.Conv1d(24, 16, kernel_size=7, padding=3),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )

        # SE block for correction patterns
        self.correction_se = SEBlock(16, reduction=2)

        # === TEMPORAL INTEGRATION ===
        # Combine impulse + correction with dilated convolutions
        self.temporal_integration = nn.Sequential(
            nn.Conv1d(48, 48, kernel_size=3, dilation=2, padding=2),  # 48 = 32 impulse + 16 correction
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(48, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 4] OHLC tensor
        Returns:
            embedding: [batch, hidden_dim] wave features
        """
        # Transpose for Conv1d: [B, T, 4] → [B, 4, T]
        x = x.permute(0, 2, 1)

        # === DETECT IMPULSE WAVES (12345) ===
        wave1 = self.wave1_detector(x)  # [B, 8, T]
        wave2 = self.wave2_detector(x)  # [B, 8, T]
        wave3 = self.wave3_detector(x)  # [B, 12, T] - EXTENDED
        wave4 = self.wave4_detector(x)  # [B, 8, T]
        wave5 = self.wave5_detector(x)  # [B, 8, T]

        # Combine impulse waves: [B, 44, T]
        impulse = torch.cat([wave1, wave2, wave3, wave4, wave5], dim=1)
        impulse = self.impulse_fusion(impulse)  # [B, 32, T]
        impulse = self.impulse_se(impulse)  # Apply attention

        # === DETECT CORRECTION WAVES (ABC) ===
        waveA = self.waveA_detector(x)  # [B, 8, T]
        waveB = self.waveB_detector(x)  # [B, 8, T]
        waveC = self.waveC_detector(x)  # [B, 8, T]

        # Combine correction waves: [B, 24, T]
        correction = torch.cat([waveA, waveB, waveC], dim=1)
        correction = self.correction_fusion(correction)  # [B, 16, T]
        correction = self.correction_se(correction)  # Apply attention

        # === COMBINE IMPULSE + CORRECTION ===
        combined = torch.cat([impulse, correction], dim=1)  # [B, 48, T]

        # Temporal integration
        output = self.temporal_integration(combined)  # [B, 48]

        # Final projection
        output = self.output_proj(output)  # [B, hidden_dim]

        return output


class ReversalPatternCNN(nn.Module):
    """
    REVERSAL PATTERN DETECTOR - H&S, Double Top/Bottom, Flags, Pennants.

    Detects:
    - HEAD & SHOULDERS (H&S):
        * Left Shoulder: Peak (15-25 candles)
        * Head: Higher peak (20-30 candles)
        * Right Shoulder: Similar to left (15-25 candles)
        * Neckline: Support line connecting troughs

    - INVERSE HEAD & SHOULDERS:
        * Same as H&S but inverted (bottoms instead of tops)

    - DOUBLE TOP:
        * Two peaks at similar levels (30-50 candles total)
        * Trough in between (~38.2% pullback)

    - DOUBLE BOTTOM:
        * Two troughs at similar levels (30-50 candles total)
        * Peak in between (~38.2% bounce)

    - FLAG PATTERNS (Continuation):
        * Sharp move (pole): 15-25 candles
        * Consolidation (flag): 8-15 candles
        * Breakout direction: Same as pole

    - PENNANT PATTERNS (Continuation):
        * Sharp move (pole): 15-25 candles
        * Converging consolidation: 10-20 candles

    Input: [batch, seq_len, 4] (OHLC) - seq_len=288
    Output: [batch, 32] (reversal pattern features)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # === HEAD & SHOULDERS DETECTOR ===
        # Wide receptive field to capture full pattern (~70 candles)
        self.hns_detector = nn.Sequential(
            nn.Conv1d(4, 12, kernel_size=71, padding=35),  # Full H&S pattern
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.Conv1d(12, 12, kernel_size=21, padding=10),  # Refine shoulders
            nn.GroupNorm(3, 12),
            nn.GELU(),
        )

        # === DOUBLE TOP/BOTTOM DETECTOR ===
        # Medium receptive field (~40-50 candles)
        self.double_pattern_detector = nn.Sequential(
            nn.Conv1d(4, 12, kernel_size=45, padding=22),  # Double pattern
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.Conv1d(12, 12, kernel_size=15, padding=7),  # Refine peaks/troughs
            nn.GroupNorm(3, 12),
            nn.GELU(),
        )

        # === FLAG DETECTOR ===
        # Pole + flag (~30-40 candles)
        self.flag_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=35, padding=17),  # Pole + flag
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv1d(8, 8, kernel_size=11, padding=5),  # Refine flag
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # === PENNANT DETECTOR ===
        # Pole + pennant (~35-45 candles)
        self.pennant_detector = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=41, padding=20),  # Pole + pennant
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv1d(8, 8, kernel_size=15, padding=7),  # Refine convergence
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )

        # === PATTERN FUSION ===
        # Combine all patterns: 12(H&S) + 12(Double) + 8(Flag) + 8(Pennant) = 40 channels
        self.pattern_fusion = nn.Sequential(
            nn.Conv1d(40, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        # SE block for pattern attention
        self.se_block = SEBlock(32, reduction=4)

        # Temporal aggregation
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 4] OHLC tensor
        Returns:
            embedding: [batch, hidden_dim] reversal pattern features
        """
        # Transpose for Conv1d: [B, T, 4] → [B, 4, T]
        x = x.permute(0, 2, 1)

        # Detect all patterns
        hns = self.hns_detector(x)              # [B, 12, T]
        double = self.double_pattern_detector(x)  # [B, 12, T]
        flag = self.flag_detector(x)            # [B, 8, T]
        pennant = self.pennant_detector(x)      # [B, 8, T]

        # Combine patterns: [B, 40, T]
        combined = torch.cat([hns, double, flag, pennant], dim=1)

        # Fuse patterns
        fused = self.pattern_fusion(combined)  # [B, 32, T]

        # Apply attention
        fused = self.se_block(fused)  # [B, 32, T]

        # Temporal pooling
        pooled = self.temporal_pool(fused)  # [B, 32]

        # Final projection
        output = self.output_proj(pooled)  # [B, hidden_dim]

        return output


class SupportResistanceCNN(nn.Module):
    """
    SUPPORT/RESISTANCE LEVEL DETECTOR with interaction tracking.

    Detects:
    - Horizontal support/resistance levels
    - Price bounces off levels (support holds)
    - Price breaks through levels (breakout)
    - Level strength: How many times tested
    - Distance from current price to key levels
    - Recent level touches (within last 50 candles)

    Input: [batch, seq_len, 4] (OHLC) - seq_len=288
    Output: [batch, 32] (S/R features)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Detect horizontal levels (wide kernel = horizontal patterns)
        self.level_detector = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=1),  # Point-wise: High/Low focus
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=31, padding=15),  # Wide: Horizontal levels
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=21, padding=10),  # Refine levels
            nn.GroupNorm(6, 24),
            nn.GELU(),
        )

        # Detect price interactions (bounces/breaks)
        self.interaction_detector = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=7, padding=3),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=5, padding=2),
            nn.GroupNorm(6, 24),
            nn.GELU(),
        )

        # SE block for level importance
        self.se_block = SEBlock(24, reduction=4)

        # Recent vs historical level detector
        self.temporal_split = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 4] OHLC tensor
        Returns:
            embedding: [batch, hidden_dim] S/R features
        """
        # Transpose for Conv1d: [B, T, 4] → [B, 4, T]
        x = x.permute(0, 2, 1)

        # Detect levels
        levels = self.level_detector(x)  # [B, 24, T]

        # Detect interactions
        interactions = self.interaction_detector(levels)  # [B, 24, T]

        # Apply attention (emphasize important levels)
        attended = self.se_block(interactions)  # [B, 24, T]

        # Temporal pooling
        pooled = self.temporal_split(attended)  # [B, 32]

        # Final projection
        output = self.output_proj(pooled)  # [B, hidden_dim]

        return output


# === IMPORT ORIGINAL MODULES ===
# We'll keep the original CNN modules and add the new pattern detectors

class SpatialOHLCCNN(nn.Module):
    """Original Spatial CNN with residual connections added"""

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 4), padding=(1, 0))
        self.norm1 = nn.GroupNorm(4, 16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(2, 0))
        self.norm2 = nn.GroupNorm(8, 32)
        self.residual_proj = nn.Conv2d(16, 32, kernel_size=1)  # Match dimensions

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(7, 1), padding=(3, 0))
        self.norm3 = nn.GroupNorm(8, 32)

        # SE block for attention
        self.se_block = SEBlock(32, reduction=4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        # Add channel dimension: [B, T, 4] → [B, 1, T, 4]
        x = x.unsqueeze(1)

        # First conv
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = torch.nn.functional.gelu(x1)

        # Second conv with residual
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        x2 = torch.nn.functional.gelu(x2 + self.residual_proj(x1))  # Residual

        # Third conv with residual
        x3 = self.conv3(x2)
        x3 = self.norm3(x3)
        x3 = torch.nn.functional.gelu(x3 + x2)  # Residual

        # SE attention
        x3 = self.se_block(x3)

        # Pool and project
        x = self.pool(x3)
        x = self.flatten(x)
        x = self.output_proj(x)

        return x


class TemporalOHLCCNN(nn.Module):
    """Original Temporal CNN - keep as is (already good)"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.temporal_paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=1, padding=0),
                nn.GroupNorm(3, 12),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=3, padding=1),
                nn.GroupNorm(3, 12),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=5, padding=2),
                nn.GroupNorm(3, 12),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=7, padding=3),
                nn.GroupNorm(3, 12),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=11, padding=5),
                nn.GroupNorm(3, 12),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(4, 12, kernel_size=15, padding=7),
                nn.GroupNorm(3, 12),
                nn.GELU(),
            ),
        ])

        self.temporal_dilated = nn.Sequential(
            nn.Conv1d(72, 64, kernel_size=3, dilation=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=8, padding=8),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(64, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        temporal_outputs = []
        for path in self.temporal_paths:
            temporal_outputs.append(path(x))

        x = torch.cat(temporal_outputs, dim=1)
        x = self.temporal_dilated(x)
        x = self.output_proj(x)

        return x


class RSI_DivergenceCNN(nn.Module):
    """
    RSI Divergence CNN with Price Context.

    Input: 3 channels (RSI + High + Low) all normalized to [-1, 1]
    Learns divergence patterns: "RSI rising while price falling" etc.
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Accept 3 channels: RSI + High + Low (all normalized [-1, 1])
        self.divergence_paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=5, padding=2),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=11, padding=5),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=21, padding=10),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=41, padding=20),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
        ])

        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        divergence_outputs = []
        for path in self.divergence_paths:
            divergence_outputs.append(path(x))

        x = torch.cat(divergence_outputs, dim=1)
        x = self.temporal_fusion(x)
        x = self.output_proj(x)

        return x


class MACD_DivergenceCNN(nn.Module):
    """
    MACD Divergence CNN with Price Context (High/Low).

    Input: 5 channels (MACD + Signal + Hist + High + Low) to detect proper divergences
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Accept 5 channels: MACD + Signal + Hist + High + Low
        self.divergence_paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(5, 8, kernel_size=5, padding=2),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(5, 8, kernel_size=11, padding=5),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(5, 8, kernel_size=21, padding=10),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(5, 8, kernel_size=41, padding=20),
                nn.GroupNorm(2, 8),
                nn.GELU(),
            ),
        ])

        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding=4),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        divergence_outputs = []
        for path in self.divergence_paths:
            divergence_outputs.append(path(x))

        x = torch.cat(divergence_outputs, dim=1)
        x = self.temporal_fusion(x)
        x = self.output_proj(x)

        return x


class TradingEnhancedExtractor(BaseFeaturesExtractor):
    """
    SIMPLIFIED Multi-Input Feature Extractor for Trading.

    ARCHITECTURE:
    - Active feature groups (8):
        1. OHLC Spatial CNN (32-dim) - Candlestick patterns
        2. OHLC Temporal CNN (64-dim) - Trend evolution
        3. RSI Divergence CNN (32-dim) - RSI pattern detection
        4. Price Context Transformer (32-dim) - Price structure
        5. Trend Indicators Transformer (32-dim) - Trend features
        6. Trading Sessions MLP (4-dim) - Session markers
        7. Account State MLP (4-dim) - Balance/equity/pnl/commission (5 features)
        8. Position Info MLP (4-dim) - Position status/leverage/distances/duration (7 features)

    - Disabled (not in environment):
        - MACD Divergence CNN
        - Momentum Oscillators MLP
        - Volume Profile features
        - VP Distribution bins
        - Pattern detection CNNs (Range, Elliott Wave, Reversal, S/R)

    Total output: 204-dim → 256-dim fused features
    """

    def __init__(self, observation_space: spaces.Dict, hidden_dim=128, **kwargs):
        out_dim = hidden_dim * 2
        super().__init__(observation_space, features_dim=out_dim)

        self.hidden_dim = hidden_dim
        self.shapes = {key: space.shape for key, space in observation_space.spaces.items()}

        # === ORIGINAL ENCODERS ===

        # 1. OHLC Spatial CNN (with residual + SE)
        self.ohlc_spatial_cnn = SpatialOHLCCNN(hidden_dim=32)

        # 2. OHLC Temporal CNN
        self.ohlc_temporal_cnn = TemporalOHLCCNN(hidden_dim=64)

        # 3. RSI Divergence CNN
        self.rsi_divergence_cnn = RSI_DivergenceCNN(hidden_dim=32)

        # === NEW PATTERN ENCODERS (DISABLED - not in environment) ===
        # # 4. Range Detection CNN
        # self.range_cnn = RangeDetectionCNN(hidden_dim=32)

        # # 5. Elliott Wave CNN
        # self.elliott_wave_cnn = ElliottWaveCNN(hidden_dim=48)

        # # 6. Reversal Pattern CNN
        # self.reversal_pattern_cnn = ReversalPatternCNN(hidden_dim=32)

        # # 7. Support/Resistance CNN
        # self.support_resistance_cnn = SupportResistanceCNN(hidden_dim=32)

        # === ORIGINAL TRANSFORMERS & MLPs ===

        # Price Context: Transformer
        self.price_projection = nn.Linear(self.shapes['price_context'][-1], 64)
        price_encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.price_transformer = nn.TransformerEncoder(
            price_encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.price_output = nn.Linear(64, 32)

        # Trend Indicators: Transformer
        # Trend Indicators: Simple MLP (reduced from Transformer to prevent over-activation)
        # Binary features (crossovers) + slopes don't need complex temporal modeling
        self.trend_encoder = nn.Sequential(
            nn.Linear(self.shapes['trend_indicators'][-1], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # Extra normalization to keep activations in check
            nn.GELU(),
        )

        # Trading Sessions: MLP
        self.session_encoder = nn.Sequential(
            nn.Linear(self.shapes['trading_sessions'][-1], 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
        )

        # Account State: MLP (5 features: equity, balance, unrealized_pnl, realized_pnl, commission)
        self.account_encoder = nn.Sequential(
            nn.Linear(self.shapes['account_state'][-1], 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
        )

        # Position Info: MLP (7 features: status, leverage, unrealized_pnl_pct, distance_to_sl, distance_to_tp, risk_reward, duration)
        self.position_encoder = nn.Sequential(
            nn.Linear(self.shapes['position_info'][-1], 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
        )

        # === TEMPORAL POOLING ===
        self.temporal_attention_32 = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, dropout=0.1, batch_first=True
        )

        self.pool_query_32 = nn.Parameter(torch.randn(1, 1, 32) * 0.01)

        # === FUSION LAYER ===
        # Active: ohlc_spatial(32) + ohlc_temporal(64) + rsi_divergence(32) +
        #         price(32) + trend(32) + session(4) + account(4) + position(4)
        # Total: 204
        combined_dim = 32 + 64 + 32 + 32 + 32 + 4 + 4 + 4

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

        # === PROCESS OHLC-BASED GROUPS ===
        ohlc_seq = obs_tensors['price_ohlc_spatial']  # [B, T, 4] - OHLC

        # Original encoders
        ohlc_spatial = self.ohlc_spatial_cnn(ohlc_seq)         # [B, 32]
        ohlc_temporal = self.ohlc_temporal_cnn(ohlc_seq)       # [B, 64]

        # === PROCESS DIVERGENCE GROUPS ===

        # RSI Divergence: RSI + High + Low (all normalized to [-1, 1] in enhanced_features.py)
        # CNN learns: "RSI rising while High falling" → Bearish divergence
        rsi_seq = obs_tensors['rsi_divergence']  # [B, T, 3] - RSI + High + Low
        rsi_divergence = self.rsi_divergence_cnn(rsi_seq)      # [B, 32]

        # Price Context
        price_seq = obs_tensors['price_context']
        price_proj = self.price_projection(price_seq)
        price_transformed = self.price_transformer(price_proj)
        price_encoded = self.price_output(price_transformed)
        price_pooled = self._pool_temporal(price_encoded, method='attention')  # [B, 32]

        # Trend Indicators: Simple MLP on last timestep (no temporal modeling needed)
        trend_seq = obs_tensors['trend_indicators']
        trend_pooled = self.trend_encoder(trend_seq[:, -1, :])  # [B, 32]

        # Trading Sessions
        session_seq = obs_tensors['trading_sessions']
        session_encoded = self.session_encoder(session_seq)
        session_pooled = session_encoded[:, -1, :]  # [B, 4]

        # Account State
        account_seq = obs_tensors['account_state']
        account_encoded = self.account_encoder(account_seq)
        account_pooled = account_encoded[:, -1, :]  # [B, 4]

        # Position Info
        position_seq = obs_tensors['position_info']
        position_encoded = self.position_encoder(position_seq)
        position_pooled = position_encoded[:, -1, :]  # [B, 4]

        # === CONCATENATE ALL EMBEDDINGS ===
        combined = torch.cat([
            ohlc_spatial,           # 32
            ohlc_temporal,          # 64
            rsi_divergence,         # 32
            price_pooled,           # 32
            trend_pooled,           # 32
            session_pooled,         # 4
            account_pooled,         # 4
            position_pooled,        # 4
        ], dim=1)  # [B, 204]

        # === FUSION ===
        fused = self.fusion(combined)  # [B, hidden_dim*2]

        return fused

    def _pool_temporal(self, x, method='last'):
        """Temporal pooling with attention support"""
        if method == 'last':
            return x[:, -1, :]
        elif method == 'mean':
            return x.mean(dim=1)
        elif method == 'max':
            return x.max(dim=1)[0]
        elif method == 'attention':
            batch_size = x.shape[0]
            feat_dim = x.shape[2]

            if feat_dim == 32:
                attention_module = self.temporal_attention_32
                query = self.pool_query_32.expand(batch_size, -1, -1)
            else:
                # Fallback to last for unsupported dimensions
                return x[:, -1, :]

            pooled, _ = attention_module(query, x, x)
            pooled = pooled.squeeze(1)

            return pooled
        else:
            raise ValueError(f"Unknown pooling method: {method}")
