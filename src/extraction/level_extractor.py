"""
Multi-timeframe Level Extractor - Extracts support/resistance levels
"""

import json
import pandas as pd
from typing import Dict, List, cast
from core.trading_types import ChartInterval, LevelInfo
from ta.middlewares.channels import channels_middleware
from ta.middlewares.levels import levels_middleware
from ta.middlewares.volume_profile import volume_profile_middleware
from ta.middlewares.zigzag import zigzag_middleware
from ta.technical_analysis import Line, Pivot, TechnicalAnalysisProcessor, AnalysisDict
from ta.timeframe_state_manager import TimeframeStateManager


class MultitimeframeLevelExtractor:
    """
    Extracts data from multiple timeframes using hardcoded extraction strategies.

    OPTIMIZATION: Uses TimeframeStateManager to cache results and avoid
    recalculating unchanged timeframes (5-10x speedup).
    """

    def __init__(self):
        """Initialize level extractor with hardcoded extraction strategies."""
        self.level_cache = {}  # Cache for extracted data

        # NEW: Timeframe state management for intelligent caching
        self.state_manager = TimeframeStateManager()
        print("✅ Level extractor initialized with timeframe caching")

    def get_extraction_config_for_timeframe(self, timeframe: str) -> Dict:
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
                    # 'zigzag': ['pivots'],  # Show all zigzag pivots for analysis
                    'levels': ['lines'],
                    'channels': ['lines'],  # Channel lines only
                }
            },
            'W': {
                'middlewares': [zigzag_middleware, levels_middleware, volume_profile_middleware],
                'extract': {
                    # 'zigzag': ['pivots'],  # Show all zigzag pivots for analysis
                    'levels': ['lines'],
                    'volume_profile_periods': ['lines']
                }
            },
            'D': {
                'middlewares': [zigzag_middleware, levels_middleware, volume_profile_middleware],
                'extract': {
                    # 'zigzag': ['pivots'],  # Show all zigzag pivots for analysis
                    'levels': ['lines'],
                    'volume_profile_periods': ['lines']
                }
            },
            '1h': {
                'middlewares': [zigzag_middleware, volume_profile_middleware],
                'extract': {
                    'zigzag': ['pivots'],
                    'volume_profile_periods': ['lines']
                }
            },
            '15m': {
                'middlewares': [zigzag_middleware, volume_profile_middleware],
                'extract': {
                    # 'zigzag': ['pivots'],
                    'volume_profile_periods': ['lines']
                }
            }
        }

        return configs.get(timeframe, {'middlewares': [], 'extract': {}})

    def extract_levels_from_dataframes(self,
                                       data_dfs: Dict[ChartInterval, pd.DataFrame],
                                       last_pivot: Pivot,
                                       live_timeframe: ChartInterval,
                                       use_log_scale: bool = True
                                       ) -> Dict[ChartInterval, Dict[str, List[Pivot | Line]]]:
        """
        OPTIMIZED: Extract raw data from pre-loaded DataFrames (much faster than loading JSON every time).

        Args:
            data_dfs: Dict mapping timeframe -> pre-loaded DataFrame
            last_pivot: Last pivot for technical analysis
            live_timeframe: The primary trading timeframe (e.g., '15m')
            use_log_scale: Whether to use logarithmic scale

        Returns:
            Dict mapping timeframe to dict with raw data: {'lines': [], 'pivots': []}
        """

        max_date = last_pivot[0]

        # ⚠️ CRITICAL: Filter all data by max_date FIRST to prevent data leakage
        # AND to ensure cache checks only see available data at this point in time
        filtered_dfs: Dict[ChartInterval, pd.DataFrame] = {}
        for timeframe_str, df_full in data_dfs.items():
            df = df_full.copy()
            # Ensure index is DatetimeIndex for proper comparison
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # Filter by max_date
            df = df[df.index <= max_date].copy()
            filtered_dfs[timeframe_str] = df

        # ⚡ OPTIMIZATION: Check which timeframes actually need recalculation
        # This prevents recalculating unchanged timeframes (e.g., Monthly data when only 15m candle arrives)
        # Pass use_log_scale to ensure cache invalidates when scale changes
        needs_update = self.state_manager.get_timeframes_needing_update(filtered_dfs, use_log_scale=use_log_scale)

        all_levels: Dict[ChartInterval, Dict[str, List]] = {}

        for timeframe_str, df_filtered in filtered_dfs.items():
            # Cast string to ChartInterval for type safety
            timeframe: ChartInterval = timeframe_str  # type: ignore  # We know these are valid ChartInterval values
            # Get hardcoded extraction config for this timeframe
            config = self.get_extraction_config_for_timeframe(timeframe)
            middlewares = config.get('middlewares', [])
            extract_config = config.get('extract', {})

            if not middlewares:
                all_levels[timeframe] = {'lines': [], 'pivots': []}
                print(f'⏭️  Skipping {timeframe} (no extraction configured)')
                continue

            if not extract_config:
                all_levels[timeframe] = {'lines': [], 'pivots': []}
                print(f'⏭️  Skipping {timeframe} (no extraction configured)')
                continue

            # ⚡ CACHE CHECK: Skip calculation if timeframe hasn't changed
            if timeframe not in needs_update:
                cached_result = self.state_manager.get_cached_result(timeframe)
                if cached_result is not None:
                    all_levels[timeframe] = cached_result
                    continue

            try:
                import time
                t_start = time.time()

                # Data is already filtered by max_date above
                t1 = time.time()
                df = df_filtered  # Already a copy from filtering step
                t_copy = time.time() - t1

                t2 = time.time()
                # No need to filter again - already done above
                t_filter = time.time() - t2

                # ⚡ OPTIMIZATION: For 1h timeframe, only look back 31 days to reduce computation
                # Recent price action is more relevant than 3 years of history
                if timeframe == '1h' and len(df) > 0:
                    from datetime import timedelta
                    lookback_date = max_date - timedelta(days=31)
                    df = df[df.index >= lookback_date].copy()

                # ⚡ OPTIMIZATION: For D timeframe, only look back 180 days to reduce computation
                # Recent price action is more relevant
                if timeframe == '15m' and len(df) > 0:
                    from datetime import timedelta
                    lookback_date = max_date - timedelta(days=7)
                    df = df[df.index >= lookback_date].copy()

                # Skip if no data available
                if len(df) == 0:
                    all_levels[timeframe] = {'lines': [], 'pivots': []}
                    continue

                # Run technical analysis with configured middlewares
                t3 = time.time()
                processor = TechnicalAnalysisProcessor(df, timeframe, last_pivot, use_log_scale)
                t_processor_init = time.time() - t3

                t4 = time.time()
                for middleware in middlewares:
                    processor.register_middleware(middleware)
                t_register = time.time() - t4

                # ⏱️ PROFILING: Time each middleware to identify bottlenecks
                t5 = time.time()
                analysis = processor.run()
                t_run = time.time() - t5

                t6 = time.time()
                # Extract data based on granular config
                levels = self._extract_data_from_analysis(analysis, extract_config)
                t_extract = time.time() - t6

                t7 = time.time()
                # ⚡ CACHE UPDATE: Store result for future use (including log_scale setting)
                self.state_manager.update_state(timeframe, df, levels, use_log_scale=use_log_scale)
                t_cache = time.time() - t7

                t_total = time.time() - t_start

                # Print detailed timing breakdown
                if t_total > 0.01:  # Only show if > 10ms
                    print(f"   ⏱️  [{timeframe}] Total: {t_total:.3f}s | "
                          f"Copy: {t_copy*1000:.0f}ms | Filter: {t_filter*1000:.0f}ms | "
                          f"Init: {t_processor_init*1000:.0f}ms | Run: {t_run*1000:.0f}ms | "
                          f"Extract: {t_extract*1000:.0f}ms | Cache: {t_cache*1000:.0f}ms")

                all_levels[timeframe] = levels

            except Exception as e:
                print(f"❌ Error processing {timeframe} data: {e}")
                import traceback
                traceback.print_exc()
                all_levels[timeframe] = {'lines': [], 'pivots': []}

        return all_levels

    # ============================================================
    # PRECOMPUTED LEVELS SUPPORT
    # ============================================================
    def load_precomputed_levels(self, parquet_path: str) -> pd.DataFrame:
        """
        Load precomputed levels parquet produced by scripts/precompute_features.py

        Expected columns:
        - datetime (str or timestamp)
        - open, high, low, close, volume, time
        - levels_json: JSON string mapping timeframe -> { 'lines': [...], 'pivots': [...] }
        - num_levels

        Returns a DataFrame indexed by datetime (pd.Timestamp)
        """
        try:

            df = pd.read_parquet(parquet_path)

            # Normalize datetime to pandas Timestamp and set as index
            if 'datetime' not in df.columns:
                raise ValueError("Precomputed file missing 'datetime' column")

            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').set_index('datetime')

            # Ensure levels_json column exists
            if 'levels_json' not in df.columns:
                raise ValueError("Precomputed file missing 'levels_json' column")

            # Quick validation: try to parse first non-null row
            sample = df['levels_json'].dropna().iloc[0]
            if isinstance(sample, str):
                try:
                    json.loads(sample)
                except Exception as e:
                    raise ValueError(f"Invalid levels_json payload: {e}")

            return df
        except Exception as e:
            print(f"❌ Error loading precomputed levels from {parquet_path}")
            raise e

    def deserialize_levels_json(self, levels_json_value) -> Dict[str, Dict[str, List]]:
        """
        Deserialize a levels_json cell into a dict[timeframe] -> {'lines': [], 'pivots': []}
        Accepts already-parsed dicts (no-op) or JSON strings.
        """
        if levels_json_value is None:
            return {}
        if isinstance(levels_json_value, dict):
            return levels_json_value  # already parsed
        if isinstance(levels_json_value, str):
            try:
                return json.loads(levels_json_value)
            except Exception:
                return {}
        return {}

    def convert_raw_to_levelinfo(self, raw_levels: Dict[str, Dict[str, List]],
                                 current_price: float) -> Dict[ChartInterval, List[LevelInfo]]:
        """
        Convert raw precomputed levels (lines/pivots) into LevelInfo structures
        expected by the LevelBasedFeatureEngineer. This is per-candle because
        distance is relative to the current price.

        Args:
            raw_levels: Dict mapping timeframe -> {'lines': [...], 'pivots': [...]} as stored in parquet
            current_price: Current candle close price

        Returns:
            Dict[ChartInterval, List[LevelInfo]]
        """
        levels_by_tf: Dict[ChartInterval, List[LevelInfo]] = {}

        for timeframe_str, payload in (raw_levels or {}).items():
            try:
                timeframe: ChartInterval = timeframe_str  # type: ignore
                lines = payload.get('lines', []) if isinstance(payload, dict) else []
                pivots = payload.get('pivots', []) if isinstance(payload, dict) else []

                # Heuristic: lines may include various sources (levels, channels, volume profile)
                # We derive multiple LevelInfo entries to preserve richness.
                tf_levels: List[LevelInfo] = []

                # Channel-specific classification if line_type hints exist
                try:
                    tf_levels.extend(self._extract_channel_levels(lines, timeframe, current_price))
                except Exception:
                    pass

                # Generic support/resistance interpretation of lines
                try:
                    tf_levels.extend(self._extract_support_resistance_levels(lines, timeframe, current_price))
                except Exception:
                    pass

                # Volume profile derived levels (POC/VAH/VAL) if present
                try:
                    tf_levels.extend(self._extract_volume_profile_levels(lines, timeframe, current_price))
                except Exception:
                    pass

                # Pivot-based levels
                try:
                    tf_levels.extend(self._extract_pivot_levels(pivots, timeframe, current_price))
                except Exception:
                    pass

                levels_by_tf[timeframe] = tf_levels
            except Exception:
                levels_by_tf[timeframe_str] = []  # type: ignore

        return levels_by_tf

    def _json_to_dataframe(self, json_data: Dict) -> pd.DataFrame:
        """Convert JSON candlestick data to DataFrame"""
        candles = json_data.get('candles', [])

        df_data = []
        for candle in candles:
            df_data.append({
                'timestamp': pd.to_datetime(candle['time'], unit='s'),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            })

        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df

    def _extract_data_from_analysis(self, analysis: AnalysisDict,
                                    extract_config: Dict[str, List[str]]) -> Dict[str, List[Line | Pivot]]:
        """
        Extract raw data from technical analysis results based on granular config.

        Args:
            analysis: Results from TechnicalAnalysisProcessor
            timeframe: Current timeframe being processed
            df: Price data DataFrame
            extract_config: Dict mapping middleware_name -> list of data types
                          e.g., {'zigzag': ['pivots'], 'levels': ['lines']}

        Returns:
            Dict with raw data: {'lines': [], 'pivots': []}
        """
        raw_data: Dict[str, List[Line | Pivot]] = {'lines': [], 'pivots': []}

        # Extract data from each middleware based on granular config
        for middleware_name, results in analysis.items():
            # Check if this middleware is in the extraction config
            if middleware_name not in extract_config:
                continue

            data_types_to_extract = extract_config[middleware_name]

            # Volume Profile levels (POC, VAH, VAL)
            if middleware_name == 'volume_profile_periods' and 'lines' in data_types_to_extract:
                if 'lines' in results:
                    raw_data['lines'].extend(cast(List[Line], results['lines']))

            # Support/Resistance levels from levels middleware
            if middleware_name == 'levels' and 'lines' in data_types_to_extract:
                if 'lines' in results:
                    raw_data['lines'].extend(cast(List[Line], results['lines']))

            # Channel levels (lines and/or pivots)
            if middleware_name == 'channels':
                if 'lines' in data_types_to_extract and 'lines' in results:
                    raw_data['lines'].extend(cast(List[Line], results['lines']))
                if 'pivots' in data_types_to_extract and 'pivots' in results:
                    raw_data['pivots'].extend(cast(List[Pivot], results['pivots']))

            # Zigzag data (lines and/or pivots)
            if middleware_name == 'zigzag':
                if 'lines' in data_types_to_extract and 'lines' in results:
                    # Extract trend lines from zigzag
                    raw_data['lines'].extend(cast(List[Line], results['lines']))

                if 'pivots' in data_types_to_extract and 'pivots' in results:
                    raw_data['pivots'].extend(cast(List[Pivot], results['pivots']))

        return raw_data

    def _extract_volume_profile_levels(self, lines: List, timeframe: ChartInterval,
                                       current_price: float) -> List[LevelInfo]:
        """Extract POC, VAH, VAL levels from volume profile data"""
        levels = []

        for line in lines:
            try:
                # Line structure: (pivot1, pivot2, line_type)
                if len(line) >= 4:
                    pivot1, pivot2, line_type, volume = line
                    if 'val_line' not in str(line_type) and 'vah_line' not in str(line_type) and 'poc_line' not in str(line_type) and 'naked_poc_line' not in str(line_type):
                        continue
                    # Extract price from pivot (timestamp, price, type)
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        levels.append(LevelInfo(
                            price=price,
                            strength=1 if timeframe == 'M' else 0.8 if timeframe == 'W' else 0.6 if timeframe == 'D' else 0.4,  # Base strength for detected levels
                            distance=distance,
                            level_type=line_type,
                            timeframe=timeframe,
                            last_test_time=pivot1[0] if len(pivot1) >= 1 else None
                        ))

            except Exception:
                continue  # Skip invalid lines

        return levels

    def _extract_support_resistance_levels(self, lines: List, timeframe: ChartInterval,
                                           current_price: float) -> List[LevelInfo]:
        """Extract support/resistance levels from lines"""
        levels = []

        for line in lines:
            try:
                # Line structure: (pivot1, pivot2, line_type)
                if len(line) >= 4:
                    pivot1, pivot2, line_type, volume = line

                    if 'line_close' not in str(line_type):
                        continue
                    # Extract price from pivot (timestamp, price, type)
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        levels.append(LevelInfo(
                            price=price,
                            strength=1 if timeframe == 'M' else 0.8 if timeframe == 'W' else 0.6 if timeframe == 'D' else 0.4,  # Base strength for detected levels
                            distance=distance,
                            level_type=line_type,
                            timeframe=timeframe,
                            last_test_time=pivot1[0] if len(pivot1) >= 1 else None
                        ))

            except Exception:
                continue  # Skip invalid lines

        return levels

    def _extract_channel_levels(self, lines: List, timeframe: ChartInterval,
                                current_price: float) -> List[LevelInfo]:
        """Extract channel boundary levels"""
        levels = []

        for line in lines:
            try:
                if len(line) >= 4:
                    pivot1, pivot2, line_type, volume = line
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if 'channel_lower_line' not in str(line_type) and 'channel_upper_line' not in str(line_type) and 'channel_middle_line' not in str(line_type):
                        continue

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        levels.append(LevelInfo(
                            price=price,
                            strength=1 if timeframe == 'M' else 0.8 if timeframe == 'W' else 0.6 if timeframe == 'D' else 0.4,  # Base strength for detected levels
                            distance=distance,
                            level_type=line_type,
                            timeframe=timeframe
                        ))

            except Exception:
                continue

        return levels

    def _extract_pivot_levels(self, pivots: List, timeframe: ChartInterval,
                              current_price: float) -> List[LevelInfo]:
        """Extract significant pivot levels"""
        levels = []

        # Only use recent pivots (last 20% of data)
        recent_pivots = pivots[-max(1, len(pivots) // 5):] if pivots else []

        for pivot in recent_pivots:
            try:
                if len(pivot) >= 3:
                    timestamp, price, pivot_type = pivot

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        level_type = f'pivot_{pivot_type}' if pivot_type else 'pivot'

                        levels.append(LevelInfo(
                            price=price,
                            strength=1 if timeframe == 'M' else 0.8 if timeframe == 'W' else 0.6 if timeframe == 'D' else 0.4,  # Base strength for detected levels
                            distance=distance,
                            level_type=level_type,
                            timeframe=timeframe,
                            last_test_time=timestamp
                        ))

            except Exception:
                continue

        return levels
