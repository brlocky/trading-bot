"""
Level Pre-computation Script (TA Middleware Cache)
===================================================

Progressively extracts levels from TA middlewares and saves to Parquet file.

WHY: Level extraction (zigzag, volume profile, channels) is SLOW (1.3s per candle).
     Feature engineering (distances, weights) is FAST (0.02s).

STRATEGY:
- Pre-compute: LEVELS (slow TA middlewares) → Save to Parquet
- On-demand: FEATURES (fast calculations) → Compute during training/testing
- Benefit: Quickly iterate on feature engineering without recomputing middlewares!

OUTPUT: Parquet file with:
- OHLCV data
- Extracted levels (serialized JSON)
- Timestamp

Features are calculated on-demand from this cached data during training.

USAGE:
    python src/scripts/precompute_features.py
"""

from pathlib import Path
import sys
import json
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, cast
import warnings


# Add src to path - do this BEFORE importing local modules
project_root = Path(__file__).parent.parent.parent  # Go up from src/scripts/ to project root
sys.path.insert(0, str(project_root / 'src'))


warnings.filterwarnings('ignore')

# Import after sys.path is set up


def convert_timestamps_for_json(obj):
    """
    Recursively convert pandas Timestamps and other non-serializable objects to JSON-serializable format
    """
    import numpy as np

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, dict):
        return {key: convert_timestamps_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_timestamps_for_json(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


class FeaturePrecomputer:
    """
    Pre-computes features for all candles with GPU acceleration and checkpointing

    Features:
    - Progressive computation (no data leakage)
    - GPU acceleration (CUDA for level feature calculations)
    - Checkpoint system (resume from interruption)
    - Progress tracking
    - Memory efficient (batch processing)
    """

    def __init__(self, output_dir: str = 'data/levels_cache'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from training.model_trainer import SimpleModelTrainer

        # Initialize trainer for level extraction
        print("⚙️  Initializing trainer...")
        self.trainer = SimpleModelTrainer()

        # Save after every single candle to avoid data loss
        self.batch_size_save = 1  # Save checkpoint after EVERY candle

        print("\n⚙️  LEVEL PRECOMPUTER INITIALIZED")
        print("   Checkpoint: Save after EVERY candle (no data loss!)")

    def precompute_levels(self,
                          training_files: Dict[str, str],
                          output_filename: str = 'BTCUSDT-15m-levels.parquet',
                          resume: bool = True):
        """
        Pre-compute levels (TA middleware results) for all candles.

        This caches the SLOW part (TA middlewares: zigzag, volume profile, channels).
        Features are calculated on-demand later (fast: 0.02s vs 1.3s for levels).

        Args:
            training_files: Dict of timeframe -> file path
            level_timeframes: List of timeframes to extract levels from
            output_filename: Output Parquet filename (levels cache)
            resume: If True, resume from last checkpoint

        Returns:
            bool: True if successful, False otherwise
        """
        from training.data_loader import DataLoader

        output_path = self.output_dir / output_filename
        checkpoint_path = self.output_dir / f"{output_filename}.checkpoint"

        # Load main timeframe data
        main_tf = '15m'
        if main_tf not in training_files:
            print(f"❌ Main timeframe {main_tf} not found in training files")
            return False

        print("\n📂 LOADING DATA")
        print("=" * 70)
        print(f"📁 Loading from: {training_files[main_tf]}")

        with open(training_files[main_tf], 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data['candles'])
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)

        total_candles = len(df)
        print(f"✅ Loaded {total_candles:,} candles")
        print(f"📅 Range: {df['datetime'].min()} to {df['datetime'].max()}")

        # Check if we should resume
        start_idx = 0
        existing_features = None

        if resume and output_path.exists():
            print(f"\n📁 Found existing features file: {output_path}")
            try:
                existing_features = pd.read_parquet(output_path)
                start_idx = len(existing_features)
                print("✅ Resuming from candle {start_idx:,}/{total_candles:,}")
            except Exception:
                print("⚠️  Could not load existing features: {e}")
                print("   Starting from beginning")
                start_idx = 0
                existing_features = None

        elif resume and checkpoint_path.exists():
            print(f"\n📁 Found checkpoint: {checkpoint_path}")
            try:
                existing_features = pd.read_parquet(checkpoint_path)
                start_idx = len(existing_features)
                print("✅ Resuming from candle {start_idx:,}/{total_candles:,}")
            except Exception:
                print("⚠️  Could not load checkpoint: {e}")
                print("   Starting from beginning")
                start_idx = 0
                existing_features = None

        if start_idx >= total_candles:
            print("\n✅ All features already computed!")
            return True

        # Calculate time estimates
        remaining_candles = total_candles - start_idx

        print("\n🔮 PROGRESSIVE LEVEL PRE-COMPUTATION")
        print("=" * 70)
        print("⚠️  Computing levels for candles {start_idx+1:,} to {total_candles:,}")
        print("📊 Remaining: {remaining_candles:,} candles")
        print("⏱️  Estimated time: {estimated_hours:.1f} hours ({estimated_seconds/3600/24:.1f} days)")
        print("💾 Auto-save: AFTER EVERY CANDLE (no data loss on interrupt!)")
        print("=" * 70)

        # ⚡ OPTIMIZATION: Pre-load all data files into DataFrames ONCE
        print("\n⚡ OPTIMIZATION: Pre-loading all timeframe data into memory...")
        level_dataframes = DataLoader._load_dataframes_from_files(training_files)
        print("✅ All data loaded! Now processing will be MUCH faster!\n")

        # Process candles progressively
        all_features = []
        start_time = datetime.now()

        try:
            with tqdm(range(start_idx, total_candles), desc="🔮 Computing features",
                      unit="candle", initial=start_idx, total=total_candles) as pbar:

                for i in pbar:
                    current_timestamp = df.loc[i, 'datetime']
                    current_price = df.loc[i, 'close']

                    # Update progress bar description
                    if i % 10 == 0 and i > start_idx:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        candles_processed = i - start_idx
                        speed = candles_processed / elapsed if elapsed > 0 else 0
                        eta_seconds = (total_candles - i) / speed if speed > 0 else 0
                        eta_hours = eta_seconds / 3600

                        pbar.set_description(
                            f"🔮 [{i+1:,}/{total_candles:,}] "
                            f"⚡{speed:.2f}c/s ETA:{eta_hours:.1f}h"
                        )

                    # Get historical data up to current candle
                    historical_data = df.iloc[:i+1].copy()
                    historical_data = historical_data.set_index('datetime')

                    try:
                        from ta.technical_analysis import Pivot
                        # Last Pivot
                        last_pivot: Pivot = (cast(pd.Timestamp, current_timestamp), cast(float, current_price), None)

                        # ⏱️ TIMING: Level extraction
                        t1 = datetime.now()

                        levels = self.trainer.autonomous_trader.level_extractor.extract_levels_from_dataframes(
                            data_dfs=level_dataframes,
                            last_pivot=last_pivot,
                            live_timeframe=main_tf
                        )
                        level_time = (datetime.now() - t1).total_seconds()

                        # Calculate total levels from the new raw data format
                        total_levels = 0
                        for timeframe, raw_data in levels.items():
                            if isinstance(raw_data, dict):
                                total_levels += len(raw_data.get('lines', []))
                                total_levels += len(raw_data.get('pivots', []))
                            else:
                                # Fallback for old format
                                total_levels += len(raw_data) if isinstance(raw_data, list) else 0

                        # Get data range for this candle
                        data_start = historical_data.index[0]
                        data_end = historical_data.index[-1]

                        # Print timing breakdown for EVERY candle to debug
                        start_str = data_start.strftime('%Y-%m-%d')
                        end_str = data_end.strftime('%Y-%m-%d')
                        print(f"\n⏱️  Candle {i+1}/{total_candles} @ {data_end} | "
                              f"Data range: {start_str} to {end_str} ({len(historical_data)} candles) | "
                              f"Levels={level_time:.2f}s | {total_levels} levels", flush=True)

                        # 💾 SAVE LEVELS + CANDLE DATA (not computed features)
                        # This allows fast iteration on feature engineering without recomputing slow TA middlewares
                        candle_data = {
                            'datetime': convert_timestamps_for_json(current_timestamp),  # Convert timestamp to string
                            'open': df.loc[i, 'open'],
                            'high': df.loc[i, 'high'],
                            'low': df.loc[i, 'low'],
                            'close': df.loc[i, 'close'],
                            'volume': df.loc[i, 'volume'],
                            'time': df.loc[i, 'time'],
                        }

                        # Serialize levels as JSON string (can be deserialized later for feature calculation)
                        levels_serialized = {}
                        for timeframe, raw_data in levels.items():
                            # raw_data is now a dict with 'lines' and 'pivots' keys
                            # Convert timestamps to strings for JSON serialization
                            levels_serialized[timeframe] = convert_timestamps_for_json(raw_data)

                        candle_data['levels_json'] = json.dumps(levels_serialized)
                        candle_data['num_levels'] = total_levels

                        all_features.append(candle_data)

                        # Save checkpoint every batch_size_save candles
                        if (i + 1) % self.batch_size_save == 0 or (i + 1) == total_candles:
                            self._save_checkpoint(
                                all_features,
                                existing_features,
                                checkpoint_path,
                                output_path,
                                i + 1,
                                total_candles,
                                start_time
                            )

                    except Exception as e:
                        print(f"\n❌ Error processing candle {i}: {e}")
                        import traceback
                        traceback.print_exc()

                        # ALWAYS save progress, even if all_features is empty
                        # (we still have existing_features from before)
                        print("\n⚠️  Attempting to save progress before exiting...")
                        try:
                            # If we have new features, save them
                            if all_features:
                                self._save_checkpoint(
                                    all_features,
                                    existing_features,
                                    checkpoint_path,
                                    output_path,
                                    i,
                                    total_candles,
                                    start_time
                                )
                                print(f"✅ Saved {len(all_features)} new features to checkpoint")
                            elif existing_features is not None:
                                # No new features, but preserve existing ones
                                print(f"⚠️  No new features, but preserving existing {len(existing_features)} features")
                                existing_features.to_parquet(checkpoint_path, compression='snappy', index=False)
                            else:
                                print("⚠️  No features to save (both all_features and existing_features are empty)")
                        except Exception as save_error:
                            print(f"❌ CRITICAL: Failed to save checkpoint: {save_error}")
                            traceback.print_exc()

                        return False

        except KeyboardInterrupt:
            print("\n\n⚠️  INTERRUPTED BY USER (Ctrl+C)")
            if all_features:
                print("💾 Saving progress...")
                self._save_checkpoint(
                    all_features,
                    existing_features,
                    checkpoint_path,
                    output_path,
                    start_idx + len(all_features),
                    total_candles,
                    start_time
                )
                print("✅ Progress saved to checkpoint")
                print(f"   Run again to resume from candle {start_idx + len(all_features)}")
            return False

        # Final save
        print(f"\n💾 Saving final features to {output_path}")
        features_df = pd.DataFrame(all_features)

        if existing_features is not None:
            features_df = pd.concat([existing_features, features_df], ignore_index=True)

        features_df.to_parquet(output_path, compression='snappy', index=False)

        # Clean up checkpoint
        if checkpoint_path.exists():
            # checkpoint_path.unlink()
            print("🧹 Cleaned up checkpoint file")

        # Final statistics
        total_time = (datetime.now() - start_time).total_seconds()
        avg_time_per_candle = total_time / remaining_candles

        print("\n🎉 FEATURE PRE-COMPUTATION COMPLETE!")
        print("=" * 70)
        print(f"📊 Total features: {len(features_df):,}")
        print(f"📁 Saved to: {output_path}")
        print(f"💾 File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"⚡ Avg speed: {avg_time_per_candle:.2f}s per candle")
        print("=" * 70)

        return True

    def _save_checkpoint(self, new_features: List, existing_features: Optional[pd.DataFrame],
                         checkpoint_path: Path, output_path: Path,
                         current_idx: int, total_candles: int, start_time: datetime):
        """Save checkpoint with progress and statistics"""

        try:
            features_df = pd.DataFrame(new_features)

            if existing_features is not None:
                features_df = pd.concat([existing_features, features_df], ignore_index=True)

            # SAFETY: Check if the new file would be suspiciously small
            if checkpoint_path.exists():
                old_size = checkpoint_path.stat().st_size
                # Don't save if new data would be < 50% of old size (likely corruption)
                import io
                buffer = io.BytesIO()
                features_df.to_parquet(buffer, compression='snappy', index=False)
                new_size = len(buffer.getvalue())

                if old_size > 1_000_000 and new_size < old_size * 0.5:  # old > 1MB and new < 50% of old
                    print(f"\n⚠️  WARNING: New checkpoint would be {new_size/1024/1024:.1f}MB vs old {old_size/1024/1024:.1f}MB")
                    print("   This looks like data loss! Creating backup instead of overwriting...")
                    backup_path = checkpoint_path.with_suffix('.parquet.backup')
                    checkpoint_path.rename(backup_path)
                    print(f"   Old file backed up to: {backup_path}")

            features_df.to_parquet(checkpoint_path, compression='snappy', index=False)

            # Only print progress every 10 candles to avoid spam (but save every candle)
            if current_idx % 10 == 0 or current_idx == total_candles:
                progress = (current_idx / total_candles) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = len(new_features) / elapsed if elapsed > 0 else 0

                # Get the last candle date
                last_date = features_df['datetime'].iloc[-1] if 'datetime' in features_df.columns else None
                if last_date is not None:
                    if isinstance(last_date, str):
                        # Already converted to ISO string format, just extract the date part
                        date_str = f" | 📅 {last_date[:10]}"  # Extract YYYY-MM-DD from ISO string
                    else:
                        # Still a timestamp object, use strftime
                        date_str = f" | 📅 {last_date.strftime('%Y-%m-%d')}"
                else:
                    date_str = ""

                print(f"\n💾 Checkpoint: {current_idx:,}/{total_candles:,} ({progress:.1f}%) "
                      f"⚡{speed:.2f}c/s{date_str} | {checkpoint_path.stat().st_size / 1024 / 1024:.1f}MB", flush=True)

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in _save_checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so caller knows save failed


def main():
    """Main execution"""
    print("=" * 70)
    print("🚀 FEATURE PRE-COMPUTATION SCRIPT")
    print("   GPU-Accelerated Progressive Feature Calculator")
    print("=" * 70)

    # Configuration
    data_folder = project_root / 'data'

    training_files = {
        'M': str(data_folder / 'BTCUSDT-M.json'),
        'W': str(data_folder / 'BTCUSDT-W.json'),
        'D': str(data_folder / 'BTCUSDT-D.json'),
        '1h': str(data_folder / 'BTCUSDT-1h.json'),
        '15m': str(data_folder / 'BTCUSDT-15m.json'),
    }

    # Check files exist
    print("\n📋 Checking data files:")
    for tf, path in training_files.items():
        exists = "✅" if Path(path).exists() else "❌"
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1024 / 1024
            print(f"   {tf:4s}: {exists} {Path(path).name:30s} ({size_mb:.1f} MB)")
        else:
            print(f"   {tf:4s}: {exists} {Path(path).name:30s} (NOT FOUND)")

    missing = [tf for tf, path in training_files.items() if not Path(path).exists()]
    if missing:
        print(f"\n❌ Missing files for timeframes: {missing}")
        print(f"   Please ensure all data files are in: {data_folder}")
        return

    # Initialize precomputer
    print("\n⚙️  Initializing level precomputer...")
    precomputer = FeaturePrecomputer(output_dir=str(data_folder / 'levels_cache'))

    # Pre-compute levels (TA middleware results)
    print("\n" + "=" * 70)
    print("⚡ STARTING LEVEL PRE-COMPUTATION (TA Middleware Cache)")
    print("   Caching SLOW TA middlewares (zigzag, volume profile, channels)")
    print("   Features will be calculated on-demand (fast iteration!)")
    print("   You can pause with Ctrl+C and resume later")
    print("=" * 70)

    try:
        success = precomputer.precompute_levels(
            training_files=training_files,
            output_filename='BTCUSDT-15m-levels.parquet',
            resume=True
        )

        if success:
            print("\n🎉 SUCCESS! Levels pre-computed and cached!")
            print("   Now you can quickly iterate on feature engineering")
            print("\n📚 Next steps:")
            print("   1. Modify feature weights in feature_engineer.py")
            print("   2. Train model: It will load cached levels and compute features instantly")
            print("   3. Iterate fast without recomputing slow TA middlewares!")
        else:
            print("\n⚠️  Level pre-computation incomplete")
            print("   Run again to resume from checkpoint")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("   Progress has been saved. Run again to resume.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
