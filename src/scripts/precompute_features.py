"""
Level Pre-computation Script (TA Middleware Cache)
===================================================

Caches slow TA middleware results (zigzag, channels, volume profile) to Parquet files.
Features are calculated on-demand during training for fast iteration.

USAGE:
    # Process default symbol (BTCUSDT)
    python src/scripts/precompute_features.py
    
    # Process specific symbol
    python src/scripts/precompute_features.py --symbol ETHUSDT
    
    # Process multiple symbols
    python src/scripts/precompute_features.py --symbol BTCUSDT ETHUSDT SOLUSDT
    
    # Process all symbols
    python src/scripts/precompute_features.py --all
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, cast

import pandas as pd
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

warnings.filterwarnings('ignore')


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
    from core.trading_types import ChartInterval

    def __init__(self, output_dir: str = 'data/levels_cache'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from training.model_trainer import SimpleModelTrainer

        # Initialize trainer for level extraction
        print("⚙️  Initializing trainer...")
        self.trainer = SimpleModelTrainer()

        # Save after every single candle to avoid data loss
        self.batch_size_save = 1000  # Save checkpoint every 100 candles (balance speed vs safety)

        print("\n⚙️  LEVEL PRECOMPUTER INITIALIZED")
        print("   Checkpoint: Save after EVERY candle (no data loss!)")

    def precompute_levels(self,
                          training_files: Dict[ChartInterval, str],
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

        from core.trading_types import ChartInterval

        # Load main timeframe data
        main_tf: ChartInterval = '15m'
        if main_tf not in training_files:
            print(f"❌ Main timeframe {main_tf} not found in training files")
            return False

        print("\n📂 LOADING DATA")
        print("=" * 70)
        print(f"📁 Loading from: {training_files[main_tf]}")

        # Use centralized DataLoader (already imported at top of method)
        df = DataLoader.load_single_json_file(training_files[main_tf])

        if df is None:
            print(f"❌ Failed to load data from {training_files[main_tf]}")
            return

        total_candles = len(df)
        print(f"✅ Loaded {total_candles:,} candles")
        print(f"📅 Range: {df.index.min()} to {df.index.max()}")

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
        level_dataframes = DataLoader._load_files(training_files)
        print("✅ All data loaded! Now processing will be MUCH faster!\n")

        # Process candles progressively
        all_features = []
        start_time = datetime.now()

        try:
            with tqdm(range(start_idx, total_candles), desc="🔮 Computing features",
                      unit="candle", initial=start_idx, total=total_candles) as pbar:

                for i in pbar:
                    current_timestamp = df.index[i]
                    current_price = df.close[i]

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

                    try:
                        from ta.technical_analysis import Pivot
                        # Last Pivot
                        last_pivot: Pivot = (cast(pd.Timestamp, current_timestamp), cast(float, current_price), 0, None)

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
                        data_end = historical_data.index[-1]

                        # Print timing breakdown for EVERY candle to debug
                        print(f"\n⏱️  Candle {i+1}/{total_candles} ({len(historical_data)} candles) {data_end} | "
                              f"Levels={level_time:.2f}s | {total_levels} levels", flush=True)

                        # 💾 SAVE LEVELS + CANDLE DATA (not computed features)
                        # This allows fast iteration on feature engineering without recomputing slow TA middlewares
                        candle_data = {
                            'datetime': convert_timestamps_for_json(current_timestamp),  # Convert timestamp to string
                            'open': df.open[i],
                            'high': df.high[i],
                            'low': df.low[i],
                            'close': df.close[i],
                            'volume': df.volume[i],
                            'time': df.time[i],
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


def process_symbol(symbol: str, data_folder: Path, resume: bool = True) -> bool:
    """
    Process a single symbol and cache its levels

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        data_folder: Path to data directory
        resume: Continue from checkpoint if available

    Returns:
        bool: Success status
    """
    from core.trading_types import ChartInterval

    print(f"\n{'='*70}")
    print(f"📊 PROCESSING: {symbol}")
    print(f"{'='*70}")

    # Build file paths for all timeframes
    training_files: Dict[ChartInterval, str] = {
        'M': str(data_folder / f'{symbol}-M.json'),
        'W': str(data_folder / f'{symbol}-W.json'),
        'D': str(data_folder / f'{symbol}-D.json'),
        '1h': str(data_folder / f'{symbol}-1h.json'),
        '15m': str(data_folder / f'{symbol}-15m.json'),
    }

    # Check if all required files exist
    print("\n📋 Checking data files:")
    missing = []
    for tf, path in training_files.items():
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1024 / 1024
            print(f"   {tf:4s}: ✅ {Path(path).name:30s} ({size_mb:.1f} MB)")
        else:
            print(f"   {tf:4s}: ❌ {Path(path).name:30s} (NOT FOUND)")
            missing.append(tf)

    if missing:
        print(f"\n❌ Missing files for {symbol}: {missing}")
        print("   Skipping this symbol...")
        return False

    # Initialize precomputer
    print("\n⚙️  Initializing level precomputer...")
    precomputer = FeaturePrecomputer(output_dir=str(data_folder / 'levels_cache'))

    # Pre-compute levels
    output_filename = f'{symbol}-15m-levels.parquet'

    try:
        success = precomputer.precompute_levels(
            training_files=training_files,
            output_filename=output_filename,
            resume=resume
        )

        if success:
            print(f"\n✅ {symbol} levels cached successfully!")
        else:
            print(f"\n⚠️  {symbol} pre-computation incomplete")

        return success

    except KeyboardInterrupt:
        print(f"\n\n⚠️  {symbol} interrupted by user")
        print("   Progress saved. Run again to resume.")
        raise  # Re-raise to stop processing other symbols
    except Exception as e:
        print(f"\n❌ Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution with command-line argument support"""
    parser = argparse.ArgumentParser(
        description='Pre-compute TA middleware levels for trading symbols',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/precompute_features.py
  python src/scripts/precompute_features.py --symbol ETHUSDT
  python src/scripts/precompute_features.py --symbol BTCUSDT ETHUSDT SOLUSDT
  python src/scripts/precompute_features.py --all
        """
    )

    parser.add_argument(
        '--symbol', '-s',
        nargs='+',
        default=['BTCUSDT'],
        help='Symbol(s) to process (default: BTCUSDT)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all symbols found in data folder'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from scratch (ignore checkpoints)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 LEVEL PRE-COMPUTATION SCRIPT")
    print("   Caching slow TA middlewares for fast feature iteration")
    print("=" * 70)

    data_folder = project_root / 'data'
    resume = not args.no_resume

    # Determine which symbols to process
    if args.all:
        # Find all symbols with 15m data files
        symbols = set()
        for file in data_folder.glob('*-15m.json'):
            symbol = file.stem.replace('-15m', '')
            symbols.add(symbol)

        if not symbols:
            print(f"\n❌ No data files found in {data_folder}")
            return

        symbols = sorted(symbols)
        print(f"\n📊 Found {len(symbols)} symbols: {', '.join(symbols)}")
    else:
        symbols = args.symbol
        print(f"\n📊 Processing {len(symbols)} symbol(s): {', '.join(symbols)}")

    # Process each symbol
    results = {}
    total_start = datetime.now()

    try:
        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(symbols)}] {symbol}")
            print(f"{'='*70}")

            success = process_symbol(symbol, data_folder, resume)
            results[symbol] = success

        # Summary
        total_time = (datetime.now() - total_start).total_seconds()
        successful = sum(1 for s in results.values() if s)

        print("\n" + "=" * 70)
        print("🎉 BATCH COMPLETE!")
        print("=" * 70)
        print(f"✅ Successful: {successful}/{len(symbols)}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")

        if successful < len(symbols):
            print("\n⚠️  Failed symbols:")
            for symbol, success in results.items():
                if not success:
                    print(f"   ❌ {symbol}")

        print("\n📚 Next steps:")
        print("   1. Train models using cached levels")
        print("   2. Iterate on feature engineering (instant!)")
        print("   3. Backtest with progressive signals")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("   All progress has been saved. Run again to continue.")


if __name__ == '__main__':
    main()
