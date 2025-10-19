"""
Feature Normalization Layer for Trading Bot

Separate, reusable normalization layer that can be used across different models
and prevents data leakage during training/testing splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import pickle
import os
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    """Base class for different normalization strategies"""

    def __init__(self):
        self.is_fitted = False
        self.stats = {}

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> 'BaseNormalizer':
        """Fit normalizer to training data"""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data using fitted parameters"""
        pass

    def fit_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Fit normalizer and transform data in one step"""
        return self.fit(data, **kwargs).transform(data, **kwargs)


class PriceRatioNormalizer(BaseNormalizer):
    """Convert prices to percentage change relative to close price"""

    def fit(self, data: np.ndarray, **kwargs) -> 'PriceRatioNormalizer':
        # No fitting required for this normalizer
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        close_prices = kwargs.get('close_prices')
        if close_prices is None:
            raise ValueError("close_prices required for price_ratio normalization")

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = (data / close_prices) - 1.0

        # Keep NaN values for early periods - they'll be handled in final step
        return ratios


class BoundedNormalizer(BaseNormalizer):
    """Normalize bounded indicators (RSI, Stochastic, etc.)"""

    def __init__(self, input_range: Tuple[float, float], target_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.input_min, self.input_max = input_range
        self.target_min, self.target_max = target_range

        # Calculate neutral value for NaN replacement
        self.neutral_input = (self.input_min + self.input_max) / 2
        self.neutral_output = (self.target_min + self.target_max) / 2

    def fit(self, data: np.ndarray, **kwargs) -> 'BoundedNormalizer':
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # Replace NaN with neutral value (e.g., RSI NaN → 50)
        filled_data = np.where(np.isnan(data), self.neutral_input, data)

        # Clip to expected range
        clipped = np.clip(filled_data, self.input_min, self.input_max)

        # Scale to target range
        input_range = self.input_max - self.input_min
        target_range = self.target_max - self.target_min

        normalized = ((clipped - self.input_min) / input_range) * target_range + self.target_min
        return normalized


class StandardNormalizer(BaseNormalizer):
    """Standard Z-score normalization with clipping and intelligent NaN handling"""

    def __init__(self, clip_range: Tuple[float, float] = (-3.0, 3.0)):
        super().__init__()
        self.clip_min, self.clip_max = clip_range

    def fit(self, data: np.ndarray, **kwargs) -> 'StandardNormalizer':
        # Calculate stats only on non-NaN values
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) > 1:
            valid_data = data[valid_mask]
            self.stats['mean'] = np.mean(valid_data)
            self.stats['std'] = np.std(valid_data) + 1e-8
        else:
            # Fallback if all data is NaN
            self.stats['mean'] = 0.0
            self.stats['std'] = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # Replace NaN with neutral value (0 after standardization)
        filled_data = np.where(np.isnan(data), self.stats['mean'], data)

        normalized = (filled_data - self.stats['mean']) / self.stats['std']

        # Clip extreme values
        return np.clip(normalized, self.clip_min, self.clip_max)


class VolumeLogNormalizer(BaseNormalizer):
    """Log normalization for volume data with NaN handling"""

    def fit(self, data: np.ndarray, **kwargs) -> 'VolumeLogNormalizer':
        # Only use non-NaN values for fitting
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) > 1:
            valid_data = data[valid_mask]
            log_data = np.log(valid_data + 1)
            self.stats['mean'] = np.mean(log_data)
            self.stats['std'] = np.std(log_data) + 1e-8
        else:
            # Fallback for all-NaN data
            self.stats['mean'] = 0.0
            self.stats['std'] = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # Replace NaN with 1 (so log(1+1) = log(2), then standardized)
        filled_data = np.where(np.isnan(data), 1.0, data)
        try:
            log_data = np.log(filled_data + 1)
            return (log_data - self.stats['mean']) / self.stats['std']
        except Exception as e:
            print(f"Error in VolumeLogNormalizer transform: {e}")
            return np.zeros_like(data)


class IdentityNormalizer(BaseNormalizer):
    """No normalization - pass through data as-is"""

    def fit(self, data: np.ndarray, **kwargs) -> 'IdentityNormalizer':
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data


class FeatureNormalizer:
    """
    Main feature normalization manager that handles multiple features
    with different normalization strategies
    """

    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initialize with normalization configuration

        Args:
            config: Dict mapping feature names to normalization types
        """
        self.config = config or {}
        self.normalizers: Dict[str, BaseNormalizer] = {}
        self.feature_columns: List[str] = []
        self.is_fitted = False

        # Define normalizer factory
        self.normalizer_factory = {
            'price_ratio': lambda: PriceRatioNormalizer(),
            'price_baseline': lambda: IdentityNormalizer(),  # Close price always 0
            'bounded_0_100': lambda: BoundedNormalizer((0, 100), (-1, 1)),
            'bounded_0_1': lambda: BoundedNormalizer((0, 1), (-1, 1)),
            'bounded_neg100_0': lambda: BoundedNormalizer((-100, 0), (-1, 1)),
            'oscillator': lambda: StandardNormalizer((-3, 3)),
            'volume_log': lambda: VolumeLogNormalizer(),
            'ratio': lambda: IdentityNormalizer(),
            'standard': lambda: StandardNormalizer((-3, 3)),
        }

    def fit(self, df: pd.DataFrame, close_prices: Optional[pd.Series] = None) -> 'FeatureNormalizer':
        """
        Fit normalizers to training data

        Args:
            df: DataFrame with features to normalize
            close_prices: Series of close prices for price_ratio normalization
        """
        self.feature_columns = list(self.config.keys())

        # Validate features exist in dataframe
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in DataFrame: {missing_features}")

        # Use close prices from dataframe if not provided
        if close_prices is None and 'close' in df.columns:
            close_prices = df['close']

        print(f"🔧 Fitting normalizers for {len(self.feature_columns)} features...")

        for feature in self.feature_columns:
            norm_type = self.config[feature]

            if norm_type not in self.normalizer_factory:
                raise ValueError(f"Unknown normalization type '{norm_type}' for {feature}, using 'standard'")

            # Create normalizer
            normalizer = self.normalizer_factory[norm_type]()

            # Fit normalizer
            feature_data = df[feature].values

            if norm_type == 'price_ratio':
                if close_prices is None:
                    raise ValueError("close_prices required for price_ratio normalization")
                normalizer.fit(feature_data, close_prices=close_prices.values)
            elif norm_type == 'price_baseline':
                # Close price is always set to 0 - no fitting needed
                normalizer.fit(feature_data)
            else:
                normalizer.fit(feature_data)

            self.normalizers[feature] = normalizer

        self.is_fitted = True
        print(f"✅ Fitted {len(self.normalizers)} normalizers")
        return self

    def transform(self, df: pd.DataFrame, close_prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform features using fitted normalizers

        Args:
            df: DataFrame with features to normalize
            close_prices: Series of close prices for price_ratio normalization

        Returns:
            DataFrame with normalized features
        """
        if not self.is_fitted:
            raise ValueError("FeatureNormalizer must be fitted before transform")

        # Use close prices from dataframe if not provided
        if close_prices is None and 'close' in df.columns:
            close_prices = df['close']

        result = df.copy()

        for feature in self.feature_columns:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in DataFrame, skipping")

            normalizer = self.normalizers[feature]

            # Standardize feature data
            feature_data = df[feature].copy()

            # Forward-fill and back-fill NaNs to avoid issues during normalization
            feature_data = feature_data.ffill().bfill().to_numpy()

            norm_type = self.config[feature]
            if norm_type == 'price_ratio':
                if close_prices is None:
                    raise ValueError("close_prices required for price_ratio normalization")
                normalized = normalizer.transform(feature_data, close_prices=close_prices.values)
            elif norm_type == 'price_baseline':
                # Close price is always 0
                normalized = np.zeros_like(feature_data)
            else:
                normalized = normalizer.transform(feature_data)

            result[feature] = normalized

        return result

    def fit_transform(self, df: pd.DataFrame, close_prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit normalizers and transform data in one step"""
        return self.fit(df, close_prices).transform(df, close_prices)

    def save(self, filepath: str):
        """Save fitted normalizers to file"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")

        # Only save essential data: config + normalizer stats
        save_data = {
            'config': self.config,
            'normalizer_stats': {
                feature: normalizer.stats
                for feature, normalizer in self.normalizers.items()
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"💾 Saved normalizer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeatureNormalizer':
        """Load fitted normalizers from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Recreate normalizer from config
        normalizer = cls(save_data['config'])

        # Rebuild normalizers and restore their stats
        for feature, norm_type in save_data['config'].items():
            new_normalizer = normalizer.normalizer_factory[norm_type]()
            new_normalizer.stats = save_data['normalizer_stats'].get(feature, {})
            new_normalizer.is_fitted = True
            normalizer.normalizers[feature] = new_normalizer

        normalizer.feature_columns = list(save_data['config'].keys())
        normalizer.is_fitted = True

        print(f"📥 Loaded normalizer from {filepath}")
        return normalizer
