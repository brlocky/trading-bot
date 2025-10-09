"""
DataLoader - Centralized data loading utilities for the trading system
"""

import json
import pandas as pd
from typing import Dict, Optional, List, Any
import warnings
import os


warnings.filterwarnings('ignore')


class DataLoader:
    """Centralized DataLoader for loading and preprocessing trading data from JSON files."""

    @staticmethod
    def load_single_json_file(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a single JSON file and convert to DataFrame with proper data types.

        Args:
            file_path: Path to the JSON file

        Returns:
            DataFrame with candle data or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            candles = data.get('candles', data) if isinstance(data, dict) else data

            if not candles:
                print(f"⚠️  No candle data found in {file_path}")
                return None

            df = pd.DataFrame(candles)

            # Standardize datetime column
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('datetime')
                # Note: Using 'datetime' as the standard timestamp column

            # Convert numeric columns with error handling
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    @staticmethod
    def load_multiple_json_files(
        file_paths: Dict[Any, str],
        exclude_keys: Optional[List[str]] = None,
        set_timestamp_index: bool = True
    ) -> Dict[Any, pd.DataFrame]:
        """
        Load multiple JSON files into DataFrames.

        Args:
            file_paths: Dict of timeframe -> file path
            exclude_keys: List of keys to exclude from loading (e.g., ['parquet_path'])
            set_timestamp_index: Whether to set timestamp as index

        Returns:
            Dict of timeframe -> DataFrame
        """
        exclude_keys = exclude_keys or ['parquet_path']
        dataframes = {}

        print("\n📂 PRE-LOADING ALL DATA FILES (one-time operation)...")

        for timeframe, file_path in file_paths.items():
            # Skip excluded keys
            if timeframe in exclude_keys:
                print(f"   Skipping {timeframe} (excluded)")
                continue

            print(f"   Loading {timeframe}...", end='', flush=True)

            df = DataLoader.load_single_json_file(file_path)
            if df is not None:
                dataframes[timeframe] = df
                print(f" ✅ {len(df):,} candles")
            else:
                print(f" ❌ Failed to load")

        print("✅ All data files loaded into memory!\n")
        return dataframes

    @staticmethod
    def load_trade_memory_json(file_path: str) -> Optional[List[dict]]:
        """
        Load trade memory JSON data (different structure than candle data).

        Args:
            file_path: Path to trade memory JSON file

        Returns:
            List of trade dictionaries or None if error
        """
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                trade_data = json.load(f)

            return trade_data if isinstance(trade_data, list) else None

        except Exception as e:
            print(f"❌ Error loading trade memory {file_path}: {e}")
            return None

    @staticmethod
    def get_symbol_files(symbol: str, data_folder: str = 'data_test') -> Dict[str, str]:
        """
        Get available JSON files for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_folder: Folder containing the data files

        Returns:
            Dict of timeframe -> file path for existing files
        """
        files = {}
        timeframes = ['15m', '1h', 'D', 'W', 'M']

        for tf in timeframes:
            path = os.path.join(data_folder, f'{symbol}-{tf}.json')
            if os.path.exists(path):
                files[tf] = path

        return files

    # Legacy method for backward compatibility
    @staticmethod
    def _load_dataframes_from_files(level_files: Dict[Any, str]) -> Dict[Any, pd.DataFrame]:
        """Legacy method - use load_multiple_json_files instead."""
        return DataLoader.load_multiple_json_files(level_files, set_timestamp_index=True)
