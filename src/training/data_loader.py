import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from core.trading_types import ChartInterval
from core.normalization_config import get_features_list


class DataLoader:
    """
    Enhanced DataLoader for the trading bot.
    Handles loading of candle data from JSON files and precomputed levels from Parquet files.
    """

    def __init__(self, data_dir: str = 'data', levels_cache_dir: str | None = None):
        """
        Initialize the DataLoader with data directories.

        Args:
            data_dir: Directory containing candle data files
            levels_cache_dir: Directory containing precomputed levels (defaults to data_dir/levels_cache)
        """
        self.data_dir = Path(data_dir)
        if levels_cache_dir is None:
            self.levels_cache_dir = self.data_dir / 'levels_cache'
        else:
            self.levels_cache_dir = Path(levels_cache_dir)

    def recalculate_higher_timeframe_indicator(self, source_df: pd.DataFrame, target_df: pd.DataFrame,
                                               source_timeframe: ChartInterval, target_tf: ChartInterval,
                                               indicator: str) -> pd.DataFrame:
        """
        Interpolate higher timeframe indicators to target timeframe using merge_asof for efficient alignment.

        Args:
            source_df: DataFrame containing source timeframe data (higher TF, fewer rows)
            target_df: DataFrame containing target timeframe data (lower TF, more rows)
            source_timeframe: The source timeframe of the data (e.g., '1h', 'D', 'W', 'M')
            target_timeframe: The target timeframe to add indicators to (e.g., '15m')
            indicator: The specific indicator to interpolate

        Returns:
            Updated target_df with interpolated higher timeframe indicator
        """

        # Define timeframe hierarchy (minutes)
        tf_hierarchy = {'15m': 15, '1h': 60, 'D': 1440, 'W': 10080, 'M': 43200}

        print(f"🔄 Interpolating {indicator} from {source_timeframe} to {target_tf}...")

        if source_timeframe == target_tf:
            raise ValueError("Source and target timeframes must be different for interpolation")

        # Ensure we're interpolating from higher to lower timeframe
        if tf_hierarchy.get(source_timeframe, 0) <= tf_hierarchy.get(target_tf, 0):
            raise ValueError(f"Source timeframe '{source_timeframe}' must be higher than target timeframe '{target_tf}'")

        if indicator not in source_df.columns:
            raise ValueError(f"Indicator '{indicator}' not found in source dataframe columns")

        try:
            print(f"  📊 Source ({source_timeframe}): {len(source_df)} rows")
            print(f"  📊 Target ({target_tf}): {len(target_df)} rows")

            # Create new column name
            new_col_name = f"{indicator}_{source_timeframe}"

            # Prepare source data with timestamp as column (not index)
            source_temp = source_df.reset_index()[[source_df.index.name or 'datetime', indicator]].copy()
            source_temp = source_temp.rename(columns={source_df.index.name or 'datetime': 'timestamp'})
            source_temp['timestamp'] = pd.to_datetime(source_temp['timestamp'])

            # Prepare target data with timestamp as column
            target_temp = target_df.reset_index()[[target_df.index.name or 'datetime']].copy()
            target_temp = target_temp.rename(columns={target_df.index.name or 'datetime': 'timestamp'})
            target_temp['timestamp'] = pd.to_datetime(target_temp['timestamp'])

            # Remove NaN values from source
            source_clean = source_temp.dropna(subset=[indicator])

            if len(source_clean) == 0:
                print(f"⚠️ No valid data for {indicator} in {source_timeframe}")
                raise ValueError(f"No valid data for {indicator} in {source_timeframe}")

            # Sort both by timestamp (required for merge_asof)
            source_clean = source_clean.sort_values('timestamp')
            target_temp = target_temp.sort_values('timestamp')

            # Use merge_asof for backward search (each target gets most recent source value)
            merged = pd.merge_asof(
                target_temp,
                source_clean[['timestamp', indicator]],
                on='timestamp',
                direction='backward'  # Each target timestamp gets the most recent source value at or before it
            )

            # Add interpolated values back to target dataframe
            target_df[new_col_name] = merged[indicator].values

        except Exception as e:
            print(f"❌ Error interpolating {source_timeframe} → {target_tf}: {e}")
            # Add column with NaN values as fallback
            target_df[f"{indicator}_{source_timeframe}"] = pd.NA
            raise e

        return target_df

    def load_data(self,
                  symbol: str,
                  timeframes: List[ChartInterval] = ['15m', '1h', 'D', 'W', 'M'],
                  target_tf: ChartInterval = '15m') -> Dict[ChartInterval, pd.DataFrame]:
        """
        Enhanced data loading with multi-timeframe indicator configuration
        """
        try:
            print(f"📥 Loading data for {symbol}...")            # Step 1: Load candle data for all timeframes
            candle_files = self._build_candle_file_config(symbol, timeframes)
            dfs = self._load_files(candle_files)

            # Step 2: Validate target timeframe
            if target_tf not in dfs:
                raise ValueError(f"Target timeframe '{target_tf}' not found in loaded dataframes")

            # Step 3: Load precomputed levels from Parquet
            levels_df = self.load_parquet(symbol, target_tf)

            # Merge levels with target timeframe candle data
            target_df = dfs[target_tf]

            # Align indices
            target_df = target_df.reindex(target_df.index)
            levels_df = levels_df.reindex(levels_df.index)

            # Copy levels_json column if exists
            if 'levels_json' not in levels_df.columns:
                raise ValueError("Levels DataFrame must contain 'levels_json' column")

            target_df['levels_json'] = levels_df['levels_json'].copy()

            # Step 4: Interpolate higher timeframe indicators to target timeframe
            if len(timeframes) > 1:

                # Get the complete feature configuration
                features_config = get_features_list()

                # Extract timeframe-specific features that need interpolation
                timeframe_features: dict = {}
                for feature_name in features_config.keys():
                    # Skip base OHLCV features
                    if feature_name in ['open', 'high', 'low', 'close', 'volume']:
                        continue

                    # Check if it's a timeframe-specific feature (has suffix like _1h, _D, etc.)
                    for tf in timeframes:
                        if tf != target_tf and feature_name.endswith(f'_{tf}'):
                            # Extract the base indicator name
                            indicator = feature_name.replace(f'_{tf}', '')

                            if tf not in timeframe_features:
                                timeframe_features[tf] = []
                            timeframe_features[tf].append(indicator)
                            break

                # Interpolate indicators for each timeframe
                for tf, indicators in timeframe_features.items():
                    if tf not in dfs:
                        print(f"⚠️ Warning: Timeframe '{tf}' not found in loaded dataframes")
                        continue

                    source_df = dfs[tf]

                    for indicator in indicators:
                        if indicator not in source_df.columns:
                            print(f"⚠️ Warning: Indicator '{indicator}' not found in {tf} dataframe")
                            continue

                        try:
                            # Copy only the indicator from the source
                            source_copy = source_df[[indicator]].copy()

                            # Expand to match target index and interpolate
                            expanded = source_copy.reindex(target_df.index.union(source_copy.index)).sort_index()
                            expanded[indicator] = expanded[indicator].interpolate(method='time')

                            # Assign interpolated values to target with proper column name
                            target_col_name = f"{indicator}_{tf}"
                            target_df[target_col_name] = expanded.reindex(target_df.index)[indicator]

                        except Exception as e:
                            print(f"❌ Error interpolating {indicator} from {tf}: {e}")
                            # Add column with NaN values as fallback
                            target_df[f"{indicator}_{tf}"] = pd.NA

            # Step 5: Store the processed target dataframe back into the dictionary
            dfs[target_tf] = target_df

            return dfs

        except Exception as e:
            raise RuntimeError(f"❌ Error loading data for {symbol}: {str(e)}")

    def _load_files(self, file_config: Dict[ChartInterval, str]) -> Dict[ChartInterval, pd.DataFrame]:
        """
        Load dataframes from JSON files for multiple timeframes.

        Args:
            file_config: Dict mapping timeframes to file paths

        Returns:
            Dict mapping timeframes to loaded dataframes
        """
        result = {}
        for tf, file_path in file_config.items():
            try:
                path = Path(file_path)
                if not path.exists():
                    raise FileNotFoundError(f"⚠️ Warning: File not found: {file_path}")

                with open(path, 'r') as f:
                    data = json.load(f)
                    df = pd.DataFrame(data['candles'])

                # Convert timestamp to datetime and set as index
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('datetime', inplace=True)
                result[tf] = df

            except Exception as e:
                raise ValueError(f"❌ Error loading {tf} data from {file_path}: {str(e)}")

        return result

    def load_parquet(self, symbol: str, timeframe: str = '15m') -> pd.DataFrame:
        """
        Load precomputed levels from Parquet file.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the levels

        Returns:
            DataFrame with levels data or None if not found
        """
        parquet_path = self.levels_cache_dir / f"{symbol}-{timeframe}-levels.parquet"
        if not parquet_path.exists():
            print(f"⚠️ Warning: Parquet cache not found: {parquet_path}")
            raise FileNotFoundError(f"Parquet cache not found: {parquet_path}")

        try:
            levels_df = pd.read_parquet(parquet_path)

            # Ensure the index is a proper DatetimeIndex
            if not isinstance(levels_df.index, pd.DatetimeIndex):
                print("🔧 Converting levels cache index to DatetimeIndex...")
                levels_df.index = pd.to_datetime(levels_df.index)

            print(f"✅ Loaded levels cache: {parquet_path}")
            print(f"📊 Shape: {levels_df.shape[0]:,} rows × {levels_df.shape[1]} columns")

            return levels_df
        except Exception as e:
            raise ValueError(f"❌ Error loading levels cache: {str(e)}")

    def _build_candle_file_config(self, symbol: str, timeframes: List[ChartInterval]) -> Dict[ChartInterval, str]:
        """
        Build configuration mapping timeframes to file paths.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes

        Returns:
            Dict mapping timeframes to file paths
        """
        return {
            tf: str(self.data_dir / f"{symbol}-{tf}.json") for tf in timeframes
        }
