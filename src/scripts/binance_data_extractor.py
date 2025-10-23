"""
Binance Data Extractor
======================

Extracts OHLCV candlestick data from Binance for multiple symbols and timeframes.
Saves data in JSON format compatible with the trading bot system.

Features:
- Multi-symbol support (20+ symbols)
- All timeframes: M, W, D, 1h, 15m
- Maximum historical data available from Binance
- Rate limit handling (1200 requests/minute, ~6 requests/symbol)
- Update mode: Merge with existing data to add only new candles
- Progress tracking with detailed logging

Usage:
    # Extract all data for all symbols
    python src/scripts/binance_data_extractor.py

    # Extract specific symbols
    python src/scripts/binance_data_extractor.py --symbols BTCUSDT ETHUSDT

    # Update existing data only
    python src/scripts/binance_data_extractor.py --update-only

Output:
    data/{SYMBOL}-{TIMEFRAME}.json
    Example: data/BTCUSDT-15m.json
"""

import sys
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


class BinanceDataExtractor:
    """
    Extracts historical candlestick data from Binance API.
    Handles rate limits, data merging, and progress tracking.
    """

    # Binance API endpoints
    BASE_URL = "https://api.binance.com"
    KLINES_ENDPOINT = "/api/v3/klines"

    # Rate limit: 1200 requests/minute = 20 requests/second
    # Safe buffer: 1 request every 0.1 seconds = 10 requests/second
    RATE_LIMIT_DELAY = 0.1  # seconds between requests

    # Binance interval mapping (API format -> Our format)
    INTERVAL_MAP = {
        '15m': '15m',
        '1h': '1h',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M'
    }

    # Maximum candles per request (Binance limit)
    MAX_LIMIT = 1000

    def __init__(self, output_dir: str = 'data'):
        """
        Initialize the extractor.

        Args:
            output_dir: Directory to save JSON files (default: 'data')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_count = 0
        self.start_time = time.time()

        print("=" * 70)
        print("🚀 BINANCE DATA EXTRACTOR")
        print("=" * 70)
        print(f"📂 Output directory: {self.output_dir.absolute()}")
        print(f"⏱️  Rate limit: {1/self.RATE_LIMIT_DELAY:.1f} requests/second")
        print()

    def _rate_limit_delay(self):
        """
        Enforce rate limiting between API requests.
        Prevents hitting Binance's 1200 requests/minute limit.
        """
        time.sleep(self.RATE_LIMIT_DELAY)
        self.request_count += 1

    def _get_klines(self, symbol: str, interval: str, start_time: Optional[int] = None,
                    end_time: Optional[int] = None, limit: int = 1000) -> List[List]:
        """
        Fetch kline/candlestick data from Binance API.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe in Binance format ('15m', '1h', '1d', '1w', '1M')
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            limit: Number of candles to fetch (max 1000)

        Returns:
            List of kline data: [[open_time, open, high, low, close, volume, ...], ...]
        """
        # Build request parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        # Make API request
        url = f"{self.BASE_URL}{self.KLINES_ENDPOINT}"
        self._rate_limit_delay()

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ API Error: {e}")
            return []

    def _fetch_all_klines(self, symbol: str, interval: str,
                          start_date: Optional[datetime] = None) -> List[Dict]:
        """
        Fetch all available historical klines for a symbol/interval.
        Handles pagination to get maximum historical data by fetching in batches.

        Strategy:
        - If start_date provided: Fetch forward from that date
        - If no start_date: Fetch backwards from present to get maximum history

        Safety: Maximum 100 batches (100,000 candles) to prevent infinite loops

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe in Binance format ('15m', '1h', '1d', '1w', '1M')
            start_date: Optional start date (for update mode)

        Returns:
            List of candle dictionaries with keys: time, open, high, low, close, volume
        """
        MAX_BATCHES = 100  # Safety limit: prevents infinite loops
        all_candles = []

        # If start_date provided, fetch forward (update mode)
        if start_date:
            current_start = int(start_date.timestamp() * 1000)
            print(f"   📥 Fetching {interval} data from {start_date.strftime('%Y-%m-%d')}...", end='', flush=True)

            for batch_num in range(MAX_BATCHES):
                klines = self._get_klines(symbol, interval, start_time=current_start, limit=self.MAX_LIMIT)

                if not klines:
                    break  # No more data available

                # Convert and add candles
                for kline in klines:
                    candle = {
                        'time': int(kline[0] / 1000),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    }
                    all_candles.append(candle)

                print(f"\r   📥 Fetching {interval} data... {len(all_candles)} candles", end='', flush=True)

                # Stop if we got less than max (reached the end)
                if len(klines) < self.MAX_LIMIT:
                    break

                # Next batch starts after last candle
                current_start = klines[-1][6] + 1

        else:
            # No start_date: Fetch backwards to get ALL available history
            print(f"   📥 Fetching ALL {interval} history...", end='', flush=True)

            # Start with most recent data
            klines = self._get_klines(symbol, interval, limit=self.MAX_LIMIT)

            if not klines:
                print(" ❌ No data")
                return []

            # Add initial batch
            for kline in klines:
                candle = {
                    'time': int(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                }
                all_candles.append(candle)

            # Now fetch backwards until we reach the beginning
            earliest_time = klines[0][0]  # Open time of first candle
            prev_earliest_time = earliest_time

            for batch_num in range(MAX_BATCHES):
                # Fetch data ending just before our earliest candle
                klines = self._get_klines(symbol, interval, end_time=earliest_time - 1, limit=self.MAX_LIMIT)

                if not klines or len(klines) == 0:
                    break  # Reached the beginning

                # Update earliest time BEFORE checking
                new_earliest_time = klines[0][0]

                # Check if we're getting the same data (no progress)
                if new_earliest_time == prev_earliest_time:
                    break  # No new data available, stop fetching

                # Convert older candles
                older_candles = []
                for kline in klines:
                    candle = {
                        'time': int(kline[0] / 1000),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    }
                    older_candles.append(candle)

                # Prepend older candles
                all_candles = older_candles + all_candles

                print(f"\r   📥 Fetching ALL {interval} history... {len(all_candles)} candles", end='', flush=True)

                # Stop if we got less than max (reached the beginning)
                if len(klines) < self.MAX_LIMIT:
                    break

                # Update for next iteration
                prev_earliest_time = earliest_time
                earliest_time = new_earliest_time

        print(" ✅")
        return all_candles

    def _load_existing_data(self, filepath: Path) -> Optional[List[Dict]]:
        """
        Load existing JSON data file if it exists.

        Args:
            filepath: Path to JSON file

        Returns:
            List of existing candles or None if file doesn't exist
        """
        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('candles', [])
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            return None

    def _merge_candles(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """
        Merge existing and new candle data, avoiding duplicates.
        Keeps only the latest version of each candle.

        Args:
            existing: List of existing candles
            new: List of new candles

        Returns:
            Merged list sorted by timestamp
        """
        # Create dict with timestamp as key (automatically handles duplicates)
        candles_dict = {candle['time']: candle for candle in existing}

        # Update with new candles (overwrites duplicates with newer data)
        for candle in new:
            candles_dict[candle['time']] = candle

        # Sort by timestamp and return as list
        merged = sorted(candles_dict.values(), key=lambda x: x['time'])
        return merged

    def _save_to_json(self, candles: List[Dict], filepath: Path, symbol: str, interval: str):
        """
        Save candles to JSON file in trading bot format.

        Format:
        {
            "symbolName": "BTCUSDT",
            "interval": "W",
            "startTime": "Mon Mar 30 2020 01:00:00 GMT+0100",
            "endTime": "Mon Apr 13 2026 00:00:00 GMT+0100",
            "candles": [
                {"time": 1234567890, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 123.45},
                ...
            ]
        }

        Args:
            candles: List of candle dictionaries
            filepath: Output file path
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe in our format (e.g., 'W', 'D', '1h', '15m')
        """

        # Get start and end times from candles
        start_timestamp = candles[0]['time']
        end_timestamp = candles[-1]['time']

        # Convert to datetime and format as required
        start_dt = datetime.fromtimestamp(start_timestamp)
        end_dt = datetime.fromtimestamp(end_timestamp)

        # Format: "Mon Mar 30 2020 01:00:00 GMT+0100"
        start_time_str = start_dt.strftime('%a %b %d %Y %H:%M:%S GMT%z')
        end_time_str = end_dt.strftime('%a %b %d %Y %H:%M:%S GMT%z')

        # Build data structure with metadata
        data = {
            'symbolName': symbol,
            'interval': interval,
            'startTime': start_time_str,
            'endTime': end_time_str,
            'candleCount': len(candles),
            'candles': candles
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"   💾 Saved to: {filepath.name}")

    def _check_existing_files(self, symbol: str) -> List[str]:
        """
        Check which files already exist for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            List of existing timeframe filenames
        """
        existing = []
        for our_interval in self.INTERVAL_MAP.values():
            filename = f"{symbol}-{our_interval}.json"
            filepath = self.output_dir / filename
            if filepath.exists():
                existing.append(filename)
        return existing

    def extract_symbol(self, symbol: str, update_mode: bool = False, force: bool = False):
        """
        Extract all timeframes for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            update_mode: If True, only fetch new data since last update
            force: If True, skip overwrite confirmation
        """
        # Check for existing files
        if not update_mode and not force:
            existing_files = self._check_existing_files(symbol)
            if existing_files:
                print(f"\n⚠️  WARNING: {len(existing_files)} existing files found for {symbol}:")
                for f in existing_files:
                    print(f"   - {f}")
                print("\n❓ This will OVERWRITE existing data. Options:")
                print("   1. Use --update-only to merge with existing data (recommended)")
                print("   2. Use --force to overwrite without confirmation")
                print("   3. Backup your data folder first")

                response = input("\n   Continue and OVERWRITE? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print(f"   ⏭️  Skipped {symbol}")
                    return None

        print(f"\n{'='*70}")
        print(f"📊 SYMBOL: {symbol}")
        print(f"{'='*70}")

        # Track statistics
        stats = {
            'symbol': symbol,
            'timeframes': {},
            'total_candles': 0,
            'new_candles': 0,
            'updated_candles': 0
        }

        # Extract each timeframe
        for binance_interval, our_interval in self.INTERVAL_MAP.items():
            print(f"\n🔄 Timeframe: {our_interval} ({binance_interval})")

            # Build output filepath
            filename = f"{symbol}-{our_interval}.json"
            filepath = self.output_dir / filename

            # Check for existing data in update mode
            existing_candles = None
            start_date = None

            if update_mode:
                existing_candles = self._load_existing_data(filepath)
                if existing_candles:
                    # Get last candle timestamp and fetch from there
                    last_timestamp = existing_candles[-1]['time']
                    start_date = datetime.fromtimestamp(last_timestamp)
                    print(f"   📅 Updating from: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print("   📅 No existing data, fetching all available")

            # Fetch data from Binance
            new_candles = self._fetch_all_klines(symbol, binance_interval, start_date)

            if not new_candles:
                print("   ⚠️  No data available")
                continue

            # Merge with existing data if in update mode
            if update_mode and existing_candles:
                original_count = len(existing_candles)
                merged_candles = self._merge_candles(existing_candles, new_candles)
                new_count = len(merged_candles) - original_count

                print(f"   📊 Existing: {original_count} | New: {new_count} | Total: {len(merged_candles)}")
                candles_to_save = merged_candles
                stats['new_candles'] += new_count
                stats['updated_candles'] += (len(new_candles) - new_count)
            else:
                print(f"   📊 Total candles: {len(new_candles)}")
                candles_to_save = new_candles
                stats['new_candles'] += len(new_candles)

            # Save to file with metadata
            self._save_to_json(candles_to_save, filepath, symbol, our_interval)

            # Get date range
            first_date = datetime.fromtimestamp(candles_to_save[0]['time']).strftime('%Y-%m-%d')
            last_date = datetime.fromtimestamp(candles_to_save[-1]['time']).strftime('%Y-%m-%d')
            print(f"   📅 Range: {first_date} to {last_date}")

            # Update stats
            stats['timeframes'][our_interval] = len(candles_to_save)
            stats['total_candles'] += len(candles_to_save)

        # Print summary for this symbol
        print(f"\n{'='*70}")
        print(f"✅ SYMBOL COMPLETE: {symbol}")
        print(f"{'='*70}")
        print(f"   Total candles across all timeframes: {stats['total_candles']:,}")
        if update_mode:
            print(f"   New candles added: {stats['new_candles']:,}")
            print(f"   Updated candles: {stats['updated_candles']:,}")
        print()

        return stats

    def extract_multiple_symbols(self, symbols: List[str], update_mode: bool = False, force: bool = False):
        """
        Extract data for multiple symbols sequentially.

        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            update_mode: If True, only fetch new data since last update
            force: If True, skip overwrite confirmations
        """
        total_start = time.time()
        all_stats = []

        print(f"\n🎯 EXTRACTING {len(symbols)} SYMBOLS")
        print(f"⚙️  Mode: {'UPDATE' if update_mode else 'FULL EXTRACTION'}")
        print(f"📊 Timeframes: {', '.join(self.INTERVAL_MAP.values())}")

        # Show warning if not in update mode and not forced
        if not update_mode and not force:
            print("\n⚠️  WARNING: Running in OVERWRITE mode!")
            print("   Existing files will be replaced. Consider using --update-only instead.")
        print()

        # Extract each symbol
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

            try:
                stats = self.extract_symbol(symbol, update_mode, force)
                if stats:
                    all_stats.append(stats)
                else:
                    print(f"   ⏭️  Skipped {symbol}")
            except KeyboardInterrupt:
                print("\n\n⚠️  Extraction interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error extracting {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Print final summary
        total_time = time.time() - total_start
        total_candles = sum(s['total_candles'] for s in all_stats)
        total_new = sum(s['new_candles'] for s in all_stats)

        print("\n" + "=" * 70)
        print("🎉 EXTRACTION COMPLETE!")
        print("=" * 70)
        print(f"   Symbols processed: {len(all_stats)}/{len(symbols)}")
        print(f"   Total candles: {total_candles:,}")
        if update_mode:
            print(f"   New candles added: {total_new:,}")
        print(f"   Total requests: {self.request_count}")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"   Average: {total_time/len(all_stats):.1f}s per symbol")
        print(f"\n📂 Data saved to: {self.output_dir.absolute()}")
        print("=" * 70)

# ============================================================
# DEFAULT SYMBOL LIST (Top 20 cryptocurrencies by market cap)
# ============================================================


DEFAULT_SYMBOLS = [
    'BTCUSDT',   # Bitcoin
    'ETHUSDT',   # Ethereum
    'BNBUSDT',   # Binance Coin
    'SOLUSDT',   # Solana
    'XRPUSDT',   # Ripple
    'ADAUSDT',   # Cardano
    'DOGEUSDT',  # Dogecoin
    'TRXUSDT',   # Tron
    'LINKUSDT',   # Chainlink
    'MATICUSDT',  # Polygon
    'DOTUSDT',    # Polkadot
    'LTCUSDT',   # Litecoin
    'AVAXUSDT',  # Avalanche
    'ATOMUSDT',  # Cosmos
    'UNIUSDT',   # Uniswap
    'XLMUSDT',   # Stellar
    'ETCUSDT',   # Ethereum Classic
    'NEARUSDT',  # Near Protocol
    'FILUSDT',   # Filecoin
    'APTUSDT',   # Aptos
]


# ============================================================
# MAIN EXECUTION
# ============================================================


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract Binance candlestick data')

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help='List of symbols to extract (default: top 20 cryptocurrencies)'
    )

    parser.add_argument(
        '--update-only',
        action='store_true',
        help='Only fetch new data since last extraction (merge mode)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip overwrite confirmation prompts (use with caution!)'
    )

    parser.add_argument(
        '--output-dir',
        default='data',
        help='Output directory for JSON files (default: data/)'
    )

    args = parser.parse_args()

    # Create extractor instance
    extractor = BinanceDataExtractor(output_dir=args.output_dir)

    # Run normal extraction
    extractor.extract_multiple_symbols(
        symbols=args.symbols,
        update_mode=args.update_only,
        force=args.force
    )

    print("\n✅ Script complete! You can now:")
    print("   1. Run training: python src/scripts/precompute_features.py")
    print("   2. Train model: Open Simple_Model_Debug.ipynb")
    print("   3. Visualize: Open Candlestick_Chart_Visualizer.ipynb")
