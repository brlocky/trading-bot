"""
Bounce Detection - Support/Resistance Level Analysis
"""

from typing import Dict, List, Tuple, Optional

from core.trading_types import LevelInfo
from extraction.level_extractor import MultitimeframeLevelExtractor


class BounceDetector:
    """
    🎯 BOUNCE DETECTOR - Support/Resistance Level Analysis
    ====================================================

    Detects when price is bouncing from key support/resistance levels:
    - Uses MultitimeframeLevelExtractor for level identification
    - Calculates bounce strength and probability
    - Integrates with trade memory for historical bounce performance
    """

    def __init__(self):
        self.level_extractor = MultitimeframeLevelExtractor()
        self.bounce_threshold = 0.5  # % distance to consider a bounce
        self.strength_multiplier = 2.0  # How much to boost signal on bounce

    def detect_bounce_opportunity(self, current_price: float, levels: Dict[str, List[LevelInfo]],
                                  signal_type: str) -> Tuple[bool, float, Optional[LevelInfo]]:
        """
        Detect if current price action represents a bounce opportunity

        Args:
            current_price: Current market price
            levels: Dictionary of levels by timeframe
            signal_type: 'BUY' or 'SELL' signal type

        Returns:
            (is_bounce, bounce_strength, level_info)
        """
        best_level = None
        best_strength = 0.0

        # Check all timeframes for relevant levels
        for timeframe, level_list in levels.items():
            for level in level_list:
                # Calculate distance to level
                distance_pct = abs(current_price - level.price) / current_price * 100

                # Skip levels that are too far away
                if distance_pct > self.bounce_threshold:
                    continue

                # Check if level type matches signal direction
                is_relevant = False
                if signal_type == 'BUY' and level.level_type in ['support', 'channel_support', 'poc', 'val']:
                    is_relevant = True
                elif signal_type == 'SELL' and level.level_type in ['resistance', 'channel_resistance', 'vah']:
                    is_relevant = True

                if not is_relevant:
                    continue

                # Calculate bounce strength based on:
                # - Level strength (how well tested)
                # - Distance (closer = stronger)
                # - Timeframe importance (higher TF = stronger)
                timeframe_weight = {'M': 3.0, 'W': 2.5, 'D': 2.0, '1h': 1.5, '15m': 1.0}.get(timeframe, 1.0)
                distance_factor = max(0.1, 1.0 - (distance_pct / self.bounce_threshold))

                strength = level.strength * timeframe_weight * distance_factor

                if strength > best_strength:
                    best_strength = strength
                    best_level = level

        # Determine if this is a strong enough bounce
        is_bounce = best_strength > 1.0  # Threshold for bounce detection

        return is_bounce, best_strength, best_level
