"""
Autonomous Trader - Main trading decision system combining multi-timeframe analysis

This is the SINGLE SOURCE OF TRUTH for all feature calculation.
All higher-level components (training, backtesting) use this for features.
"""

import pandas as pd
import joblib
from datetime import datetime
from typing import Dict

from extraction.level_extractor import MultitimeframeLevelExtractor
from extraction.feature_engineer import LevelBasedFeatureEngineer
from memory.trade_memory import TradeMemoryManager
from detection.bounce_detector import BounceDetector

from ta.middlewares import (
    zigzag_middleware,
    levels_middleware,
    channels_middleware,
    volume_profile_middleware
)


class AutonomousTrader:
    """
    Main autonomous trading system that combines multi-timeframe analysis
    with level-based feature engineering to make trading decisions.

    This is the SINGLE SOURCE OF TRUTH for feature calculation:
    - 45 TA features (RSI, MACD, BB, etc.)
    - 52 Level features (distance, strength, etc.)
    - 10 Memory features (win rate, consecutive, etc.)
    = 107 total features

    Features are cached intelligently:
    - TA features: Cached by data range (recalc when new candles)
    - Level features: Cached by day (recalc at midnight UTC)
    - Memory features: Always fresh (no cache)
    """

    # Middleware configurations per timeframe (for TA feature calculation)
    # These run during _calculate_ta_features() to add extra TA columns
    MIDDLEWARE_CONFIG = {
        '15m': [zigzag_middleware, volume_profile_middleware, channels_middleware],
        '1h': [zigzag_middleware, channels_middleware],
        'D': [zigzag_middleware, levels_middleware, channels_middleware],
        'W': [zigzag_middleware, levels_middleware, channels_middleware],
        'M': [zigzag_middleware, levels_middleware, channels_middleware]
    }

    @staticmethod
    def get_extraction_config_for_timeframe(timeframe: str) -> Dict:
        """
        Hardcoded extraction configuration per timeframe.
        Defines which middlewares to run and what data to extract from each.

        Args:
            timeframe: '15m', '1h', 'D', 'W', 'M'

        Returns:
            Dict with:
            - middlewares: List of middlewares to run
            - extract: Dict mapping middleware_name -> list of data types
                      e.g., {'zigzag': ['pivots'], 'levels': ['lines']}

        Middleware outputs:
            - zigzag: ['lines', 'pivots']
            - levels: ['lines']
            - channels: ['lines', 'pivots']
            - volume_profile_periods: ['lines']
        """
        configs = {
            'M': {
                'middlewares': [zigzag_middleware, levels_middleware, channels_middleware],
                'extract': {
                    'zigzag': ['pivots'],
                    'levels': ['lines'],
                    'channels': ['lines'],
                }
            },
            'W': {
                'middlewares': [zigzag_middleware, levels_middleware, channels_middleware],
                'extract': {
                    'zigzag': ['pivots'],
                    'levels': ['lines'],
                    'channels': ['lines'],
                }
            },
            'D': {
                'middlewares': [zigzag_middleware, levels_middleware, channels_middleware],
                'extract': {
                    'zigzag': ['pivots'],
                    'levels': ['lines'],
                    'channels': ['lines'],
                }
            },
            '1h': {
                'middlewares': [zigzag_middleware, channels_middleware],
                'extract': {
                    'zigzag': ['pivots', 'lines'],
                    'channels': ['lines'],
                }
            },
            '15m': {
                'middlewares': [zigzag_middleware, levels_middleware, channels_middleware, volume_profile_middleware],
                'extract': {
                    'zigzag': ['pivots'],
                    'channels': ['lines'],
                    'volume_profile_periods': ['lines']
                }
            }
        }

        return configs.get(timeframe, {'middlewares': [], 'extract': {}})

    def __init__(self):
        """
        Initialize AutonomousTrader.

        Uses hardcoded configurations:
        - MIDDLEWARE_CONFIG: Controls which middlewares run during TA feature calculation
        - get_extraction_config_for_timeframe(): Returns what to extract per timeframe

        To customize, edit the class variables and static method above.
        """
        self.level_extractor = MultitimeframeLevelExtractor()
        self.feature_engineer = LevelBasedFeatureEngineer()

        # Memory and bounce detection systems
        self.trade_memory = TradeMemoryManager()
        self.bounce_detector = BounceDetector()

        # Middleware configuration (for TA features)
        self.middleware_config = self.MIDDLEWARE_CONFIG

        # Model state
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.current_levels = {}
        self.last_update = None

        # Caching state for features
        self._ta_cache = None
        self._ta_cache_key = None
        self._level_cache = None
        self._level_cache_date = None

    # ============================================================
    # FEATURE CALCULATION - SINGLE SOURCE OF TRUTH
    # ============================================================

    def get_all_features(
        self,
        data: pd.DataFrame,
        levels: Dict,
        current_date: pd.Timestamp,
        timeframe: str = '15m',
        use_log_scale: bool = True,

    ) -> pd.DataFrame:
        """
        Calculate ALL features for the given data.
        This is the SINGLE SOURCE OF TRUTH for features across training, backtesting, and live trading.

        Features are cached intelligently:
        - TA features (Group 1): Cached by data range, recalculated when data changes
        - Level features (Group 2): Cached per day, recalculated when day changes
        - Memory features: Always calculated fresh (they're lightweight)

        Args:
            data: OHLCV DataFrame with datetime index
            levels: Dict of levels by timeframe (from MultitimeframeLevelExtractor)
            timeframe: Timeframe identifier (e.g., '15m', '1h', 'D')
            use_log_scale: Whether to use logarithmic scale for TA features
            current_date: Current timestamp (defaults to last candle timestamp)

        Returns:
            DataFrame with all features:
            - 45 TA features (RSI, MACD, BB, etc.)
            - 52 Level features (distance, strength, etc.)
            - 10 Memory features (win rate, consecutive, etc.)
            = 107 total features
        """
        if current_date is None:
            current_date = data.index[-1]

        print("\n" + "="*70)
        print("🔮 FEATURE CALCULATION (AutonomousTrader - Single Source of Truth)")
        print("="*70)
        start_time = datetime.now()

        # 1. Calculate TA features (Group 1 - cached by data range)
        ta_features = self._calculate_ta_features(data, timeframe, use_log_scale)

        # 2. Calculate Level features (Group 2 - cached by day, checks day change)
        level_features = self._calculate_level_features(data, levels, current_date)

        # 3. Calculate Memory features (always fresh)
        memory_features = self._calculate_memory_features(data)

        # 4. Combine all
        all_features = pd.concat([ta_features, level_features, memory_features], axis=1)

        total_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ TOTAL FEATURES: {len(all_features.columns)}")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print("="*70 + "\n")

        return all_features

    def _calculate_ta_features(self, data: pd.DataFrame, timeframe: str, use_log_scale: bool) -> pd.DataFrame:
        """
        Calculate technical analysis features using existing TA processor.

        Caching strategy:
        - Cache key = (timeframe, start_date, end_date, num_candles)
        - Recalculates when data range changes (new candles added)
        - This is FAST for backtesting (calculate once for entire dataset)
        """
        start_time = datetime.now()

        # Create cache key
        start_date = data.index[0].strftime('%Y%m%d')
        end_date = data.index[-1].strftime('%Y%m%d')
        cache_key = f"ta_{timeframe}_{start_date}_{end_date}_{len(data)}"

        # Check cache
        if self._ta_cache is not None and self._ta_cache_key == cache_key:
            print(f"✅ Using cached TA features ({cache_key})")
            return self._ta_cache

        print(f"🔄 Calculating TA features ({cache_key})...")

        # Add traditional TA indicators (RSI, MACD, BB, etc.) using TA-Lib
        from indicator_utils import add_progressive_indicators

        df_with_indicators = add_progressive_indicators(data.copy())

        # Extract only the indicator columns (exclude OHLCV and metadata)
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        metadata_columns = ['levels_json']  # Exclude non-feature columns from parquet
        exclude_columns = base_columns + metadata_columns
        indicator_columns = [col for col in df_with_indicators.columns if col not in exclude_columns]

        ta_features = df_with_indicators[indicator_columns].bfill().ffill()

        # Cache it
        self._ta_cache = ta_features
        self._ta_cache_key = cache_key

        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ TA features calculated and cached: {len(ta_features.columns)} features ({elapsed_time:.2f}s)")
        return ta_features

    def _calculate_level_features(self, data: pd.DataFrame, levels: Dict, current_date: datetime) -> pd.DataFrame:
        """
        Calculate level-based features for all candles.

        No caching - data changes every iteration in progressive training.
        """
        start_time = datetime.now()
        total_levels = sum(len(lvls) for lvls in levels.values())
        print(f"🔄 Calculating level features: {len(data):,} candles with {total_levels:,} levels...")

        from extraction.feature_engineer import LevelBasedFeatureEngineer
        engineer = LevelBasedFeatureEngineer()

        # Only use parallel processing if we have MANY candles (overhead for small/medium batches)
        # Progressive training = small batches, so use higher threshold
        PARALLEL_THRESHOLD = 1000  # Use parallel only if >= 1000 candles

        if len(data) >= PARALLEL_THRESHOLD:
            print(f"   🚀 Using parallel processing (large batch: {len(data)} candles)...")

            from joblib import Parallel, delayed
            import multiprocessing

            # Use all available cores
            n_jobs = min(8, multiprocessing.cpu_count())  # Cap at 8 to avoid overhead
            print(f"   Using {n_jobs} CPU cores...")

            # Parallel processing with progress
            feature_rows = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(engineer.create_level_features)(
                    float(row['close']),
                    float(row['volume']),
                    levels
                )
                for idx, row in data.iterrows()
            )
        else:
            print(f"   ⚡ Using sequential processing (small batch: {len(data)} candles)...")

            # Sequential processing (faster for small batches due to no overhead)
            feature_rows = []
            for idx, row in data.iterrows():
                features = engineer.create_level_features(
                    float(row['close']),
                    float(row['volume']),
                    levels
                )
                feature_rows.append(features)

        level_features = pd.DataFrame(feature_rows, index=data.index)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Level features calculated: {len(level_features.columns)} features ({elapsed_time:.2f}s)")
        return level_features

    def _calculate_memory_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate memory-based features for all candles.

        Memory features are always calculated fresh (no cache) because:
        - They reflect current trade history
        - Calculation is very fast (< 1ms)
        - They need to be updated after each trade
        """
        start_time = datetime.now()
        feature_rows = []

        # Get current memory stats (same for all candles in this batch)
        recent_perf = self.trade_memory.get_recent_performance()
        bounce_perf = self.trade_memory.get_bounce_performance()
        consecutive = self.trade_memory.get_consecutive_performance()

        for idx, row in data.iterrows():
            memory_features = {
                'win_rate_7d': recent_perf['win_rate'],
                'avg_pnl_7d': recent_perf['avg_pnl'],
                'bounce_win_rate': bounce_perf['bounce_win_rate'],
                'consecutive_wins': max(0, consecutive),
                'consecutive_losses': max(0, -consecutive),
                'total_trades': len(self.trade_memory.trades),
                'recent_trade_count': len([t for t in self.trade_memory.trades if t.is_recent(7)]),
                'avg_hold_time': recent_perf.get('avg_hold_time', 0),
                'max_drawdown': recent_perf.get('max_drawdown', 0),
                'sharpe_ratio': recent_perf.get('sharpe_ratio', 0),
            }
            feature_rows.append(memory_features)

        memory_features_df = pd.DataFrame(feature_rows, index=data.index)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Memory features calculated: {len(memory_features_df.columns)} features ({elapsed_time:.3f}s)")

        return memory_features_df
