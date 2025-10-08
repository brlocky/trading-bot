"""
Multi-timeframe Level Extractor - Extracts support/resistance levels
"""

import json
import pandas as pd
from typing import Dict, List, cast
from core.trading_types import ChartInterval, LevelInfo
from ta.technical_analysis import Line, Pivot, TechnicalAnalysisProcessor, AnalysisDict, VolumeProfileLine


class MultitimeframeLevelExtractor:
    """
    Extracts data from multiple timeframes using hardcoded extraction strategies.

    Uses AutonomousTrader.get_extraction_config_for_timeframe() to determine:
    - Which middlewares to run per timeframe
    - What data to extract (levels, lines, pivots, vp)
    """

    def __init__(self):
        """Initialize level extractor with hardcoded extraction strategies."""
        self.level_cache = {}  # Cache for extracted data

    def _resample_and_merge_timeframes(self,
                                       data_dfs: Dict[ChartInterval, pd.DataFrame],
                                       live_timeframe: ChartInterval,
                                       max_date: pd.Timestamp) -> Dict[ChartInterval, pd.DataFrame]:
        """
        Resample the live timeframe data to higher timeframes and merge with existing data.
        This prevents using outdated higher timeframe candles.

        Example:
        - If live_timeframe='15m' and max_date='2023-05-20 14:30'
        - Resample 15m -> 1h, D, W, M up to 14:30
        - Merge with existing 1h, D, W, M data (keeps historical data, updates current incomplete candle)

        Args:
            data_dfs: Dict of timeframe -> DataFrame
            live_timeframe: The primary trading timeframe (smallest granularity)
            max_date: Current datetime

        Returns:
            Updated dict with resampled higher timeframes
        """
        if live_timeframe not in data_dfs:
            print(f"⚠️  Live timeframe {live_timeframe} not found in data_dfs")
            return data_dfs

        live_df = data_dfs[live_timeframe].copy()

        # Filter to max_date
        live_df = live_df[live_df.index <= max_date]

        if len(live_df) == 0:
            print(f"⚠️  No data in live timeframe {live_timeframe} up to {max_date}")
            return data_dfs

        print(f"   📊 Live data: {len(live_df)} candles ({live_df.index[0]} to {live_df.index[-1]})")

        # Define resampling rules: live_timeframe -> higher timeframes
        # Only resample to timeframes that exist in ChartInterval
        resample_rules = {
            '15m': {'1h': '1H', 'D': '1D', 'W': '1W', 'M': '1ME'},  # M = Month End
            '1h': {'D': '1D', 'W': '1W', 'M': '1ME'},
            'D': {'W': '1W', 'M': '1ME'},
            'W': {'M': '1ME'},
        }

        if live_timeframe not in resample_rules:
            print(f"   ℹ️  No resampling rules for {live_timeframe}")
            return data_dfs

        rules = resample_rules[live_timeframe]
        updated_dfs = data_dfs.copy()

        for target_tf, pandas_rule in rules.items():
            try:
                # Resample live data to target timeframe
                resampled = live_df.resample(pandas_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                if len(resampled) == 0:
                    continue

                print(f"   ✅ Resampled {live_timeframe} -> {target_tf}: {len(resampled)} candles")

                # Merge with existing data if available
                if target_tf in data_dfs and len(data_dfs[target_tf]) > 0:
                    existing_df = data_dfs[target_tf].copy()

                    # Keep historical data (before resampled range)
                    historical = existing_df[existing_df.index < resampled.index[0]]

                    # Combine: historical + resampled (resampled overwrites overlapping period)
                    merged = pd.concat([historical, resampled])
                    merged = merged[~merged.index.duplicated(keep='last')]  # Keep last (resampled) for duplicates
                    merged = merged.sort_index()

                    updated_dfs[target_tf] = merged
                    print(f"      📌 Merged with existing: {len(historical)} historical + {len(resampled)} resampled = {len(merged)} total")
                else:
                    # No existing data, use resampled only
                    updated_dfs[target_tf] = resampled
                    print(f"      📌 Using resampled data only: {len(resampled)} candles")

            except Exception as e:
                print(f"   ❌ Error resampling {live_timeframe} -> {target_tf}: {e}")
                continue

        print("   ✅ Resampling complete!\n")
        return updated_dfs

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
        from trading.autonomous_trader import AutonomousTrader
        max_date = last_pivot[0]
        # ============================================================
        # STEP 1: RESAMPLE LIVE TIMEFRAME TO HIGHER TIMEFRAMES
        # ============================================================
        # This ensures all timeframes reflect the same current market state
        # and prevents using outdated data from incomplete higher timeframe candles
        print(f"\n🔄 Resampling {live_timeframe} data to higher timeframes...")
        data_dfs = self._resample_and_merge_timeframes(data_dfs, live_timeframe, max_date)

        all_levels: Dict[ChartInterval, Dict[str, List]] = {}

        for timeframe_str, df_full in data_dfs.items():
            # Cast string to ChartInterval for type safety
            timeframe: ChartInterval = timeframe_str  # type: ignore  # We know these are valid ChartInterval values
            # Get hardcoded extraction config for this timeframe
            config = AutonomousTrader.get_extraction_config_for_timeframe(timeframe)
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

            try:
                # ⚠️ CRITICAL: Filter data by max_date to prevent data leakage
                df = df_full.copy()  # Work with a copy

                df = df[df.index <= max_date].copy()

                # ⚡ OPTIMIZATION: For 1h timeframe, only look back 7 days to reduce computation
                # Recent price action is more relevant than 3 years of history
                if timeframe == '1h' and len(df) > 0:
                    from datetime import timedelta
                    lookback_date = max_date - timedelta(days=7)
                    df = df[df.index >= lookback_date].copy()
                    print(f"   ⚡ 1h optimization: Limited to last 7 days ({len(df)} candles)")

                # ⚡ OPTIMIZATION: For D timeframe, only look back 180 days to reduce computation
                # Recent price action is more relevant
                if timeframe == 'D' and len(df) > 0:
                    from datetime import timedelta
                    lookback_date = max_date - timedelta(days=180)
                    df = df[df.index >= lookback_date].copy()
                    print(f"   ⚡ D optimization: Limited to last 180 days ({len(df)} candles)")

                # ⚡ OPTIMIZATION: For D timeframe, only look back 180 days to reduce computation
                # Recent price action is more relevant
                if timeframe == '15m' and len(df) > 0:
                    from datetime import timedelta
                    lookback_date = max_date - timedelta(days=7)
                    df = df[df.index >= lookback_date].copy()
                    print(f"   ⚡ 15m optimization: Limited to last 7 days ({len(df)} candles)")

                # Skip if no data available
                if len(df) == 0:
                    all_levels[timeframe] = {'lines': [], 'pivots': []}
                    continue

                # 🔍 DEBUG: Show what we're processing
                print(f"🔍 [{timeframe}] Processing {len(df)} candles (max_date={max_date})")
                if len(df) > 0:
                    print(f"   Data range: {df.index[0]} to {df.index[-1]}")

                # Run technical analysis with configured middlewares
                processor = TechnicalAnalysisProcessor(df, timeframe, last_pivot, use_log_scale)

                for middleware in middlewares:
                    processor.register_middleware(middleware)

                analysis = processor.run()

                # Extract data based on granular config
                levels = self._extract_data_from_analysis(analysis, extract_config)
                all_levels[timeframe] = levels

                if len(df) % 100 == 0:
                    print(f"✅ Extracted {len(levels)} items from {timeframe} timeframe")

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

                # De-duplicate by price within small tolerance to avoid overweighting identical levels
                if tf_levels:
                    uniq: List[LevelInfo] = []
                    seen_prices = set()
                    for lvl in tf_levels:
                        key = round(lvl.price, 5)
                        if key not in seen_prices:
                            seen_prices.add(key)
                            uniq.append(lvl)
                    tf_levels = uniq

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
                                    extract_config: Dict[ChartInterval, List[str]]) -> Dict[str, List[Line | Pivot]]:
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
                if len(line) >= 3:
                    pivot1, pivot2, line_type = line

                    # Extract price from pivot (timestamp, price, type)
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        # Determine if support or resistance based on price relative to current
                        level_type = 'support' if price < current_price else 'resistance'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.8,  # Base strength for detected levels
                            distance=distance,
                            level_type=level_type,
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
                if len(line) >= 3:
                    pivot1, pivot2, line_type = line

                    # Extract price from pivot (timestamp, price, type)
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        # Determine if support or resistance based on price relative to current
                        level_type = 'support' if price < current_price else 'resistance'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.8,  # Base strength for detected levels
                            distance=distance,
                            level_type=level_type,
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
                if len(line) >= 3:
                    pivot1, pivot2, line_type = line
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        # Channel type determines level type
                        if 'upper' in str(line_type):
                            level_type = 'channel_resistance'
                        elif 'lower' in str(line_type):
                            level_type = 'channel_support'
                        else:
                            level_type = 'channel_middle'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.7,  # Channel levels have good strength
                            distance=distance,
                            level_type=level_type,
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
                            strength=0.6,  # Pivot levels have moderate strength
                            distance=distance,
                            level_type=level_type,
                            timeframe=timeframe,
                            last_test_time=timestamp
                        ))

            except Exception:
                continue

        return levels
