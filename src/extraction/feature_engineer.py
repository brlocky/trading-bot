"""
Level-Based Feature Engineering - Creates features from price-level interactions
"""

import numpy as np
from typing import Dict, List

from src.core.trading_types import LevelInfo
from src.ta.technical_analysis import ChartInterval


class LevelBasedFeatureEngineer:
    """Creates features based on price interaction with key levels"""

    def __init__(self):
        self.proximity_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]  # Distance thresholds in %

    def create_level_features(self, current_price: float, current_volume: float,
                              levels: Dict[ChartInterval, List[LevelInfo]]) -> Dict[str, float]:
        """
        Create features based on current price interaction with levels

        Args:
            current_price: Current market price
            current_volume: Current volume
            levels: Dict of levels by timeframe

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # Flatten all levels
        all_levels = []
        for timeframe_levels in levels.values():
            all_levels.extend(timeframe_levels)

        if not all_levels:
            return self._get_default_features()

        # Sort levels by distance from current price
        all_levels.sort(key=lambda x: x.distance)

        # Proximity features
        features.update(self._create_proximity_features(current_price, all_levels))

        # Level strength features
        features.update(self._create_strength_features(all_levels))

        # Support/Resistance balance
        features.update(self._create_support_resistance_balance(current_price, all_levels))

        # Volume context
        features['current_volume'] = current_volume
        features['volume_normalized'] = min(current_volume / 1000000, 10.0)  # Normalize volume

        # Timeframe importance
        features.update(self._create_timeframe_features(levels))

        return features

    def _create_proximity_features(self, current_price: float,
                                   levels: List[LevelInfo]) -> Dict[str, float]:
        """Create features based on proximity to levels"""
        features = {}

        # Find closest levels in each direction
        closest_support = None
        closest_resistance = None

        for level in levels:
            if level.price < current_price and (not closest_support or level.distance < closest_support.distance):
                closest_support = level
            elif level.price > current_price and (not closest_resistance or level.distance < closest_resistance.distance):
                closest_resistance = level

        # Distance to closest levels
        features['distance_to_support'] = closest_support.distance if closest_support else 100.0
        features['distance_to_resistance'] = closest_resistance.distance if closest_resistance else 100.0

        # Support/resistance strength
        features['support_strength'] = closest_support.strength if closest_support else 0.0
        features['resistance_strength'] = closest_resistance.strength if closest_resistance else 0.0

        # Count levels within proximity thresholds
        for threshold in self.proximity_thresholds:
            count = sum(1 for level in levels if level.distance <= threshold)
            features[f'levels_within_{threshold}pct'] = count

        return features

    def _create_strength_features(self, levels: List[LevelInfo]) -> Dict[str, float]:
        """Create features based on level strength"""
        features = {}

        if not levels:
            return {'avg_level_strength': 0.0, 'max_level_strength': 0.0}

        strengths = [level.strength for level in levels]
        features['avg_level_strength'] = np.mean(strengths)
        features['max_level_strength'] = np.max(strengths)

        # Strength by level type
        level_types = {}
        for level in levels:
            if level.level_type not in level_types:
                level_types[level.level_type] = []
            level_types[level.level_type].append(level.strength)

        for level_type, type_strengths in level_types.items():
            features[f'{level_type}_strength'] = np.mean(type_strengths)

        return features

    def _create_support_resistance_balance(self, current_price: float,
                                           levels: List[LevelInfo]) -> Dict[str, float]:
        """Create features representing support/resistance balance"""
        features = {}

        support_levels = [l for l in levels if l.price < current_price]
        resistance_levels = [l for l in levels if l.price > current_price]

        features['support_count'] = len(support_levels)
        features['resistance_count'] = len(resistance_levels)
        features['support_resistance_ratio'] = (
            len(support_levels) / max(len(resistance_levels), 1)
        )

        # Weighted strength balance
        support_strength = sum(l.strength / max(l.distance, 0.1) for l in support_levels)
        resistance_strength = sum(l.strength / max(l.distance, 0.1) for l in resistance_levels)

        features['weighted_support_strength'] = support_strength
        features['weighted_resistance_strength'] = resistance_strength
        features['strength_balance'] = (
            support_strength / max(resistance_strength, 0.1)
        )

        return features

    def _create_timeframe_features(self, levels: Dict[ChartInterval, List[LevelInfo]]) -> Dict[str, float]:
        """Create features based on timeframe importance"""
        features = {}

        timeframe_weights = {'M': 3.0, 'W': 2.0, 'D': 1.0}  # Monthly > Weekly > Daily

        for timeframe, weight in timeframe_weights.items():
            if timeframe in levels:
                count = len(levels[timeframe])
                avg_strength = np.mean([l.strength for l in levels[timeframe]]) if levels[timeframe] else 0.0

                features[f'{timeframe}_level_count'] = count
                features[f'{timeframe}_avg_strength'] = avg_strength
                features[f'{timeframe}_weighted_importance'] = count * avg_strength * weight

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when no levels are available"""
        features = {}

        # Set all features to neutral/zero values
        features.update({
            'distance_to_support': 100.0,
            'distance_to_resistance': 100.0,
            'support_strength': 0.0,
            'resistance_strength': 0.0,
            'support_count': 0,
            'resistance_count': 0,
            'support_resistance_ratio': 1.0,
            'weighted_support_strength': 0.0,
            'weighted_resistance_strength': 0.0,
            'strength_balance': 1.0,
            'avg_level_strength': 0.0,
            'max_level_strength': 0.0,
            'current_volume': 1000000,
            'volume_normalized': 1.0
        })

        # Proximity features
        for threshold in self.proximity_thresholds:
            features[f'levels_within_{threshold}pct'] = 0

        # Timeframe features
        for timeframe in ['M', 'W', 'D']:
            features[f'{timeframe}_level_count'] = 0
            features[f'{timeframe}_avg_strength'] = 0.0
            features[f'{timeframe}_weighted_importance'] = 0.0

        return features
