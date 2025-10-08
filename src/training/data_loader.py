"""
Simple Model Trainer - Main training interface for the trading system
"""

import json
import pandas as pd
from typing import Dict
import warnings

from core.trading_types import ChartInterval

warnings.filterwarnings('ignore')


class DataLoader:
    """ DataLoader for loading and preprocessing trading data from JSON files."""
    @staticmethod
    def _load_dataframes_from_files(level_files: Dict[ChartInterval, str]) -> Dict[ChartInterval, pd.DataFrame]:
        """
        OPTIMIZED: Load all JSON files into DataFrames once (called at start of pre-computation).

        Args:
            level_files: Dict of timeframe -> file path

        Returns:
            Dict of timeframe -> DataFrame
        """

        dataframes = {}
        print("\n📂 PRE-LOADING ALL DATA FILES (one-time operation)...")

        for timeframe, file_path in level_files.items():
            try:
                print(f"   Loading {timeframe}...", end='', flush=True)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                # Convert to DataFrame
                candles = json_data.get('candles', [])
                df_data = []
                for candle in candles:
                    df_data.append({
                        'timestamp': pd.to_datetime(candle['time'], unit='s'),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volume']),
                        'time': candle['time']
                    })

                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                dataframes[timeframe] = df

                print(f" ✅ {len(df):,} candles")

            except Exception as e:
                print(f" ❌ Error: {e}")
                raise e

        print("✅ All data files loaded into memory!\n")
        return dataframes
