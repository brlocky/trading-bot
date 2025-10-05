"""
🔗 MULTI-TIMEFRAME DATA PIPELINE - Autonomous Trading Data Management
====================================================================
Manages data flow between multiple timeframes for autonomous trading:

1. Level extraction from higher timeframes (M, W, D)
2. Real-time level updates and caching
3. Intraday data integration
4. Feature engineering pipeline
5. Model prediction pipeline

This pipeline ensures efficient data processing and level management
for the autonomous trading system.
"""

import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path
import threading
import time
from dataclasses import dataclass, asdict

from src.autonomous_trader import (
    AutonomousTrader, MultitimeframeLevelExtractor,
    LevelBasedFeatureEngineer, TradingSignal
)

warnings.filterwarnings('ignore')


@dataclass
class MarketSnapshot:
    """Market data snapshot for decision making"""
    timestamp: datetime
    symbol: str
    timeframe: str
    price: float
    volume: float
    high: float
    low: float
    open: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LevelUpdate:
    """Level update information"""
    timestamp: datetime
    timeframe: str
    levels_count: int
    significant_changes: bool
    update_reason: str


class DataPipelineManager:
    """
    Manages multi-timeframe data pipeline for autonomous trading
    """

    def __init__(self, data_directory: str = "data", cache_directory: str = "cache"):
        self.data_directory = Path(data_directory)
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)

        # Components
        self.level_extractor = MultitimeframeLevelExtractor()
        self.feature_engineer = LevelBasedFeatureEngineer()
        self.autonomous_trader = AutonomousTrader()

        # Data management
        self.current_levels = {}
        self.level_cache = {}
        self.last_level_update = {}
        self.market_snapshots = []

        # Configuration
        self.supported_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT']
        self.higher_timeframes = ['M', 'W', 'D']
        self.trading_timeframes = ['1h', '4h']
        self.level_update_intervals = {
            'M': timedelta(days=7),    # Update monthly levels weekly
            'W': timedelta(days=1),    # Update weekly levels daily
            'D': timedelta(hours=4)    # Update daily levels every 4 hours
        }

        # Threading for background updates
        self.update_thread = None
        self.stop_updates = False

    def initialize_pipeline(self, symbol: str = 'BTCUSDT',
                            load_models: bool = True) -> bool:
        """
        Initialize the data pipeline for a specific symbol

        Args:
            symbol: Trading symbol to initialize
            load_models: Whether to load trained models

        Returns:
            Success status
        """
        print(f"🚀 Initializing data pipeline for {symbol}")

        try:
            # Load models if requested
            if load_models:
                model_file = self.cache_directory / 'autonomous_trader_models.joblib'
                if model_file.exists():
                    success = self.autonomous_trader.load_models(str(model_file))
                    if not success:
                        print("⚠️ Failed to load models, continuing without models")
                else:
                    print("⚠️ No trained models found, pipeline will work without predictions")

            # Initialize level extraction for the symbol
            data_files = self._get_data_files(symbol)
            missing_files = [tf for tf, file in data_files.items() if not Path(file).exists()]

            if missing_files:
                print(f"⚠️ Missing data files for timeframes: {missing_files}")
                print("   Pipeline will work with available data")

            # Extract initial levels
            available_files = {tf: file for tf, file in data_files.items()
                               if Path(file).exists()}

            if available_files:
                print("🔍 Extracting initial levels...")
                levels = self.level_extractor.extract_levels_from_data(available_files)
                self.current_levels[symbol] = levels
                self.last_level_update[symbol] = datetime.now()

                # Cache levels
                self._cache_levels(symbol, levels)

                print(f"✅ Initialized with {sum(len(lvls) for lvls in levels.values())} levels")
            else:
                print("❌ No data files available for level extraction")
                self.current_levels[symbol] = {}

            # Update autonomous trader with levels
            if symbol in self.current_levels:
                self.autonomous_trader.current_levels = self.current_levels[symbol]
                self.autonomous_trader.last_update = self.last_level_update[symbol]

            print(f"✅ Pipeline initialized for {symbol}")
            return True

        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            return False

    def _get_data_files(self, symbol: str) -> Dict[str, str]:
        """Get data file paths for a symbol"""
        data_files = {}
        for timeframe in self.higher_timeframes:
            filename = f"{symbol}-{timeframe}.json"
            filepath = self.data_directory / filename
            data_files[timeframe] = str(filepath)
        return data_files

    def _cache_levels(self, symbol: str, levels: Dict[Any, Any]):
        """Cache extracted levels to disk"""
        try:
            cache_file = self.cache_directory / f"{symbol}_levels_cache.json"

            # Convert levels to serializable format
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'levels_summary': {}
            }

            for timeframe, level_list in levels.items():
                cache_data['levels_summary'][timeframe] = {
                    'count': len(level_list),
                    'types': list(set(level.level_type for level in level_list)),
                    'price_range': {
                        'min': min(level.price for level in level_list) if level_list else None,
                        'max': max(level.price for level in level_list) if level_list else None
                    }
                }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            print(f"⚠️ Warning: Failed to cache levels: {e}")

    def process_market_data(self, market_data: Dict[str, Any]) -> MarketSnapshot:
        """
        Process incoming market data into standardized format

        Args:
            market_data: Raw market data dictionary

        Returns:
            MarketSnapshot object
        """
        # Handle different input formats
        if 'close' in market_data:
            price = float(market_data['close'])
        elif 'price' in market_data:
            price = float(market_data['price'])
        else:
            raise ValueError("Market data must contain 'close' or 'price' field")

        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol=market_data.get('symbol', 'BTCUSDT'),
            timeframe=market_data.get('timeframe', '1h'),
            price=price,
            volume=float(market_data.get('volume', 1000000)),
            high=float(market_data.get('high', price)),
            low=float(market_data.get('low', price)),
            open=float(market_data.get('open', price))
        )

        # Store snapshot
        self.market_snapshots.append(snapshot)

        # Keep only recent snapshots (last 1000)
        if len(self.market_snapshots) > 1000:
            self.market_snapshots = self.market_snapshots[-1000:]

        return snapshot

    def get_trading_signal(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Get trading signal for current market conditions

        Args:
            market_data: Current market data

        Returns:
            TradingSignal with action and reasoning
        """
        # Process market data
        snapshot = self.process_market_data(market_data)

        # Check if levels need updating
        self._check_and_update_levels(snapshot.symbol)

        # Create additional technical features
        additional_features = self._calculate_additional_features(snapshot)

        # Get trading signal from autonomous trader
        signal = self.autonomous_trader.make_trading_decision(
            snapshot.price,
            snapshot.volume,
            additional_features
        )

        return signal

    def _calculate_additional_features(self, snapshot: MarketSnapshot) -> Dict[str, float]:
        """Calculate additional technical features from recent snapshots"""
        features = {}

        # Get recent snapshots for the same symbol
        recent_snapshots = [s for s in self.market_snapshots[-100:]
                            if s.symbol == snapshot.symbol]

        if len(recent_snapshots) < 5:
            return self._get_default_additional_features()

        # Price-based features
        prices = [s.price for s in recent_snapshots]
        volumes = [s.volume for s in recent_snapshots]

        try:
            # Moving averages
            features['sma_5'] = np.mean(prices[-5:])
            features['sma_10'] = np.mean(prices[-10:]) if len(prices) >= 10 else features['sma_5']
            features['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else features['sma_5']

            # Price momentum
            if len(prices) >= 5:
                features['momentum_5'] = (prices[-1] / prices[-5] - 1) * 100
            else:
                features['momentum_5'] = 0.0

            # Volatility
            if len(prices) >= 2:
                returns = np.diff(prices) / np.array(prices[:-1])
                features['volatility'] = np.std(returns) * 100
            else:
                features['volatility'] = 2.0

            # Volume features
            features['volume_sma_5'] = np.mean(volumes[-5:])
            features['volume_ratio'] = volumes[-1] / features['volume_sma_5'] if features['volume_sma_5'] > 0 else 1.0

            # Price relative to moving averages
            current_price = prices[-1]
            features['price_vs_sma5'] = (current_price / features['sma_5'] - 1) * 100
            features['price_vs_sma10'] = (current_price / features['sma_10'] - 1) * 100
            features['price_vs_sma20'] = (current_price / features['sma_20'] - 1) * 100

        except Exception as e:
            print(f"⚠️ Warning: Error calculating additional features: {e}")
            return self._get_default_additional_features()

        return features

    def _get_default_additional_features(self) -> Dict[str, float]:
        """Default additional features when calculation fails"""
        return {
            'sma_5': 50000.0,
            'sma_10': 50000.0,
            'sma_20': 50000.0,
            'momentum_5': 0.0,
            'volatility': 2.0,
            'volume_sma_5': 1000000.0,
            'volume_ratio': 1.0,
            'price_vs_sma5': 0.0,
            'price_vs_sma10': 0.0,
            'price_vs_sma20': 0.0
        }

    def _check_and_update_levels(self, symbol: str):
        """Check if levels need updating and update if necessary"""
        if symbol not in self.last_level_update:
            return

        current_time = datetime.now()
        last_update = self.last_level_update[symbol]

        # Check each timeframe
        needs_update = False
        for timeframe, interval in self.level_update_intervals.items():
            if current_time - last_update > interval:
                needs_update = True
                break

        if needs_update:
            print(f"🔄 Updating levels for {symbol}")
            self._update_levels_background(symbol)

    def _update_levels_background(self, symbol: str):
        """Update levels in background thread"""
        try:
            data_files = self._get_data_files(symbol)
            available_files = {tf: file for tf, file in data_files.items()
                               if Path(file).exists()}

            if available_files:
                new_levels = self.level_extractor.extract_levels_from_data(available_files)

                # Update current levels
                old_count = sum(len(lvls) for lvls in self.current_levels.get(symbol, {}).values())
                new_count = sum(len(lvls) for lvls in new_levels.values())

                self.current_levels[symbol] = new_levels
                self.last_level_update[symbol] = datetime.now()

                # Update autonomous trader
                self.autonomous_trader.current_levels = new_levels
                self.autonomous_trader.last_update = self.last_level_update[symbol]

                # Cache updated levels
                self._cache_levels(symbol, new_levels)

                print(f"✅ Updated levels for {symbol}: {old_count} → {new_count}")

        except Exception as e:
            print(f"❌ Error updating levels for {symbol}: {e}")

    def start_background_updates(self, update_interval: int = 3600):
        """
        Start background thread for periodic level updates

        Args:
            update_interval: Update interval in seconds (default: 1 hour)
        """
        if self.update_thread and self.update_thread.is_alive():
            print("⚠️ Background updates already running")
            return

        self.stop_updates = False
        self.update_thread = threading.Thread(
            target=self._background_update_worker,
            args=(update_interval,),
            daemon=True
        )
        self.update_thread.start()
        print(f"🔄 Started background updates (interval: {update_interval}s)")

    def stop_background_updates(self):
        """Stop background update thread"""
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("🛑 Stopped background updates")

    def _background_update_worker(self, interval: int):
        """Background worker for periodic updates"""
        while not self.stop_updates:
            try:
                # Update levels for all initialized symbols
                for symbol in self.current_levels.keys():
                    self._check_and_update_levels(symbol)

                # Sleep in small chunks to allow for quick stopping
                for _ in range(interval):
                    if self.stop_updates:
                        break
                    time.sleep(1)

            except Exception as e:
                print(f"❌ Error in background update worker: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'initialized_symbols': list(self.current_levels.keys()),
            'background_updates_running': (self.update_thread and
                                           self.update_thread.is_alive() and
                                           not self.stop_updates),
            'market_snapshots_count': len(self.market_snapshots),
            'models_loaded': bool(self.autonomous_trader.models),
            'levels_summary': {}
        }

        # Add levels summary
        for symbol, levels in self.current_levels.items():
            status['levels_summary'][symbol] = {
                'timeframes': list(levels.keys()),
                'total_levels': sum(len(lvls) for lvls in levels.values()),
                'last_update': self.last_level_update.get(symbol, datetime.min).isoformat()
            }

        return status

    def export_pipeline_data(self, output_file: str):
        """Export pipeline data for analysis"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'pipeline_status': self.get_pipeline_status(),
            'recent_snapshots': [s.to_dict() for s in self.market_snapshots[-100:]],
            'current_levels_summary': {}
        }

        # Add current levels summary
        for symbol, levels in self.current_levels.items():
            export_data['current_levels_summary'][symbol] = {}
            for timeframe, level_list in levels.items():
                export_data['current_levels_summary'][symbol][timeframe] = [
                    {
                        'price': level.price,
                        'strength': level.strength,
                        'distance': level.distance,
                        'level_type': level.level_type,
                        'timeframe': level.timeframe,
                        'last_test_time': level.last_test_time.isoformat() if level.last_test_time else None
                    }
                    for level in level_list[:10]  # Export top 10 levels per timeframe
                ]

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"📤 Exported pipeline data to {output_file}")


def demo_pipeline():
    """Demonstrate the data pipeline"""
    print("🔗 Multi-Timeframe Data Pipeline Demo")
    print("=" * 50)

    # Initialize pipeline
    pipeline = DataPipelineManager()

    # Initialize for BTCUSDT
    success = pipeline.initialize_pipeline('BTCUSDT', load_models=False)

    if not success:
        print("❌ Pipeline initialization failed")
        return

    # Get pipeline status
    status = pipeline.get_pipeline_status()
    print("📊 Pipeline Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")

    # Simulate market data processing
    print("\n🎯 Testing trading signal generation...")

    sample_market_data = {
        'symbol': 'BTCUSDT',
        'close': 65000.0,
        'volume': 2500000,
        'high': 65200.0,
        'low': 64800.0,
        'open': 64950.0
    }

    # Get trading signal
    signal = pipeline.get_trading_signal(sample_market_data)

    print("📈 Trading Signal:")
    print(f"   Action: {signal.action.value}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Entry: ${signal.entry_price:,.2f}" if signal.entry_price else "   Entry: N/A")
    print(f"   Stop Loss: ${signal.stop_loss:,.2f}" if signal.stop_loss else "   Stop Loss: N/A")
    print(f"   Take Profit: ${signal.take_profit:,.2f}" if signal.take_profit else "   Take Profit: N/A")
    print(f"   Reasoning: {signal.reasoning}")

    # Export pipeline data
    pipeline.export_pipeline_data('pipeline_demo_export.json')

    print("\n✅ Pipeline demo completed successfully!")


if __name__ == "__main__":
    demo_pipeline()
