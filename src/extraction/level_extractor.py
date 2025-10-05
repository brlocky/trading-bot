"""
Multi-timeframe Level Extractor - Extracts support/resistance levels
"""

import pandas as pd
import json
from typing import Dict, List

from src.core.trading_types import LevelInfo
from src.ta.technical_analysis import TechnicalAnalysisProcessor, AnalysisDict, ChartInterval
from src.ta.middlewares.zigzag import zigzag_middleware
from src.ta.middlewares.volume_profile import volume_profile_middleware
from src.ta.middlewares.channels import channels_middleware
from src.ta.middlewares.levels import levels_middleware


class MultitimeframeLevelExtractor:
    """Extracts key levels from multiple timeframes"""

    def __init__(self):
        self.higher_timeframes: List[ChartInterval] = ['M', 'W', 'D']  # Monthly, Weekly, Daily
        self.level_cache = {}  # Cache for extracted levels

    def extract_levels_from_data(self, data_files: Dict[str, str],
                                 use_log_scale: bool = True) -> Dict[ChartInterval, List[LevelInfo]]:
        """
        Extract key levels from multiple timeframe data files

        Args:
            data_files: Dict mapping timeframe to file path {'M': 'path/to/monthly.json', ...}
            use_log_scale: Whether to use logarithmic scale

        Returns:
            Dict mapping timeframe to list of LevelInfo objects
        """
        all_levels = {}

        for timeframe, file_path in data_files.items():
            if timeframe not in self.higher_timeframes:
                continue

            try:
                # Load and process data
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                # Convert to DataFrame
                df = self._json_to_dataframe(json_data)

                # Run technical analysis
                processor = TechnicalAnalysisProcessor(df, timeframe, use_log_scale)
                processor.register_middleware(zigzag_middleware)
                processor.register_middleware(volume_profile_middleware)
                processor.register_middleware(channels_middleware)
                processor.register_middleware(levels_middleware)

                analysis = processor.run()

                # Extract levels from analysis
                levels = self._extract_levels_from_analysis(analysis, timeframe, df)
                all_levels[timeframe] = levels

                print(f"✅ Extracted {len(levels)} levels from {timeframe} timeframe")

            except Exception as e:
                print(f"❌ Error processing {timeframe} data: {e}")
                all_levels[timeframe] = []

        return all_levels

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

    def _extract_levels_from_analysis(self, analysis: AnalysisDict,
                                      timeframe: ChartInterval,
                                      df: pd.DataFrame) -> List[LevelInfo]:
        """Extract LevelInfo objects from technical analysis results"""
        levels = []
        current_price = df['close'].iloc[-1]

        # Extract levels from different middlewares
        for middleware_name, results in analysis.items():

            # Volume Profile levels (POC, VAH, VAL)
            if 'lines' in results and middleware_name == 'volume_profile_periods':
                levels.extend(self._extract_volume_profile_levels(
                    results['lines'], timeframe, current_price
                ))

            # Support/Resistance levels from levels middleware
            if 'lines' in results and middleware_name == 'levels':
                levels.extend(self._extract_support_resistance_levels(
                    results['lines'], timeframe, current_price
                ))

            # Channel levels
            if 'lines' in results and middleware_name == 'channels':
                levels.extend(self._extract_channel_levels(
                    results['lines'], timeframe, current_price
                ))

            # Pivot levels from zigzag
            if 'pivots' in results and middleware_name == 'zigzag':
                levels.extend(self._extract_pivot_levels(
                    results['pivots'], timeframe, current_price
                ))

        return levels

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

            except Exception as e:
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

            except Exception as e:
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

            except Exception as e:
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

            except Exception as e:
                continue

        return levels
