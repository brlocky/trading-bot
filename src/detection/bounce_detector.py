"""
Bounce Detection - Support/Resistance Level Analysis
"""

import numpy as np
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

    def find_tested_support_resistance(self, current_price: float, levels: List[LevelInfo],
                                       direction: str = 'support', min_tests: int = 2) -> Optional[LevelInfo]:
        """
        Find the nearest tested support or resistance level

        Args:
            current_price: Current market price
            levels: List of all levels
            direction: 'support' (below price) or 'resistance' (above price)
            min_tests: Minimum number of tests (strength) required

        Returns:
            LevelInfo of tested level or None
        """
        # Filter levels by direction and strength
        if direction == 'support':
            candidates = [lvl for lvl in levels
                          if lvl.price < current_price
                          and lvl.strength >= min_tests
                          and lvl.level_type in ['LL', 'HL', 'POC', 'VAL']]
        else:  # resistance
            candidates = [lvl for lvl in levels
                          if lvl.price > current_price
                          and lvl.strength >= min_tests
                          and lvl.level_type in ['HH', 'LH', 'VAH']]

        if not candidates:
            return None

        # Return closest level by distance
        return min(candidates, key=lambda x: x.distance)

    def calculate_labeling_quality(self, current_price: float, levels: List[LevelInfo],
                                   signal_type: str, rsi: float = 50.0) -> float:
        """
        Calculate quality score for training labels (0-1 scale)

        Args:
            current_price: Current market price
            levels: List of all levels
            signal_type: 'BUY' or 'SELL'
            rsi: Current RSI value

        Returns:
            Quality score 0.0 to 1.0
        """
        quality_components = []

        # Component 1: Level proximity (30% weight)
        if signal_type == 'BUY':
            support = self.find_tested_support_resistance(current_price, levels, 'support')
            if support and support.distance < 1.0:  # Within 1%
                proximity_score = max(0.3, 1.0 - support.distance)
                quality_components.append(proximity_score)
            else:
                quality_components.append(0.3)
        else:  # SELL
            resistance = self.find_tested_support_resistance(current_price, levels, 'resistance')
            if resistance and resistance.distance < 1.0:
                proximity_score = max(0.3, 1.0 - resistance.distance)
                quality_components.append(proximity_score)
            else:
                quality_components.append(0.3)

        # Component 2: RSI confluence (20% weight)
        if signal_type == 'BUY':
            if rsi < 30:
                quality_components.append(1.0)
            elif rsi < 40:
                quality_components.append(0.7)
            else:
                quality_components.append(0.3)
        else:  # SELL
            if rsi > 70:
                quality_components.append(1.0)
            elif rsi > 60:
                quality_components.append(0.7)
            else:
                quality_components.append(0.3)

        # Component 3: Level strength (20% weight)
        if signal_type == 'BUY':
            support = self.find_tested_support_resistance(current_price, levels, 'support')
            strength_score = min((support.strength / 5.0), 1.0) if support else 0.3
            quality_components.append(strength_score)
        else:
            resistance = self.find_tested_support_resistance(current_price, levels, 'resistance')
            strength_score = min((resistance.strength / 5.0), 1.0) if resistance else 0.3
            quality_components.append(strength_score)

        # Return weighted average
        return float(np.mean(quality_components)) if quality_components else 0.5

    def find_pivot_stop_loss(self, current_price: float, levels: List[LevelInfo],
                             signal_type: str) -> Optional[float]:
        """
        Find appropriate stop loss based on pivot levels

        Args:
            current_price: Current market price
            levels: List of all levels
            signal_type: 'BUY' or 'SELL'

        Returns:
            Stop loss price or None
        """
        if signal_type == 'BUY':
            # SL below nearest LL/HL pivot
            pivots = [lvl for lvl in levels
                      if lvl.price < current_price
                      and lvl.level_type in ['LL', 'HL']]
        else:  # SELL
            # SL above nearest HH/LH pivot
            pivots = [lvl for lvl in levels
                      if lvl.price > current_price
                      and lvl.level_type in ['HH', 'LH']]

        if not pivots:
            return None

        nearest_pivot = min(pivots, key=lambda x: x.distance)
        return nearest_pivot.price

    def find_volume_profile_target(self, current_price: float, levels: List[LevelInfo],
                                   signal_type: str) -> Optional[float]:
        """
        Find take profit target based on volume profile levels

        Args:
            current_price: Current market price
            levels: List of all levels
            signal_type: 'BUY' or 'SELL'

        Returns:
            Take profit price or None
        """
        if signal_type == 'BUY':
            # TP at VAH or POC above price
            targets = [lvl for lvl in levels
                       if lvl.price > current_price
                       and lvl.level_type in ['VAH', 'POC']]
        else:  # SELL
            # TP at VAL or POC below price
            targets = [lvl for lvl in levels
                       if lvl.price < current_price
                       and lvl.level_type in ['VAL', 'POC']]

        if not targets:
            return None

        nearest_target = min(targets, key=lambda x: x.distance)
        return nearest_target.price
