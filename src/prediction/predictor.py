"""
Simple Model Predictor - Generates predictions using trained models
"""

import pandas as pd
import os
import json
from typing import Dict, Optional

from src.trading.autonomous_trader import AutonomousTrader


class SimpleModelPredictor:
    """
    Simple prediction system for testing trained models
    """

    def __init__(self, model_trainer):
        """
        Args:
            model_trainer: SimpleModelTrainer instance with loaded model
        """
        self.model_trainer = model_trainer
        self.trader = AutonomousTrader()

    def _load_json_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load JSON data and convert to DataFrame"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['time'], unit='s')

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df.sort_values('datetime').reset_index(drop=True)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _get_symbol_files(self, symbol: str, data_folder: str = 'data_test') -> Dict[str, str]:
        """Get available files for a symbol"""
        files = {}
        for tf in ['15m', '1h', 'D', 'W', 'M']:
            path = f'{data_folder}/{symbol}-{tf}.json'
            if os.path.exists(path):
                files[tf] = path
                continue
            alt_path = f'{data_folder}/{symbol}-{tf} (1).json'
            if os.path.exists(alt_path):
                files[tf] = alt_path
        return files

    def generate_predictions(self, symbol: str, num_candles: int = 200,
                             data_folder: str = 'data_test',
                             buy_threshold: float = 0.25, sell_threshold: float = 0.25,
                             aggressive_threshold: float = 0.20) -> Optional[pd.DataFrame]:
        """
        Generate predictions for a symbol with adjustable aggressiveness

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            num_candles: Number of recent candles to analyze
            data_folder: Folder containing test data
            buy_threshold: Minimum probability for BUY signals (lower = more signals)
            sell_threshold: Minimum probability for SELL signals (lower = more signals)
            aggressive_threshold: Even lower threshold for borderline cases
        """
        if not self.model_trainer.is_trained:
            print("❌ No trained model loaded!")
            return None

        print(f"🎯 Generating predictions for {symbol}")

        # Get test files
        test_files = self._get_symbol_files(symbol, data_folder)
        if not test_files:
            print(f"❌ No data files found for {symbol}")
            return None

        print(f"📁 Found files: {list(test_files.keys())}")

        # Load main data (prefer 15m, fallback to 1h)
        main_timeframe = '15m' if '15m' in test_files else '1h'
        test_data = self._load_json_data(test_files[main_timeframe])

        if test_data is None:
            print("❌ Could not load test data")
            return None

        print(f"✅ Loaded {len(test_data)} candles ({main_timeframe})")

        # Extract levels
        level_files = {tf: path for tf, path in test_files.items() if tf in ['M', 'W', 'D']}
        if not level_files:
            print("❌ No level files found")
            return None

        success = self.trader.update_levels(level_files, force_update=True)
        if not success:
            print("❌ Level extraction failed")
            return None

        total_levels = sum(len(levels) for levels in self.trader.current_levels.values())
        print(f"✅ Extracted {total_levels} levels")

        # Get recent data for predictions
        recent_data = test_data.tail(num_candles).copy()
        print(f"🔍 Processing {len(recent_data)} recent candles")

        # Generate predictions
        signals = []
        model_data = self.model_trainer.model_data
        trained_model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_columns = model_data['feature_columns']
        model_type = model_data['model_type']

        for i, (_, row) in enumerate(recent_data.iterrows()):
            try:
                if i % 50 == 0:
                    print(f"   Processing {i+1}/{len(recent_data)}...")

                # Create features using the same method as training
                features = self.trader.feature_engineer.create_level_features(
                    float(row['close']), float(row['volume']), self.trader.current_levels
                )

                # ADD REAL MEMORY FEATURES FOR TESTING (same as training)
                if self.model_trainer.enable_memory_features:
                    recent_perf = self.model_trainer.trade_memory.get_recent_performance()
                    bounce_perf = self.model_trainer.trade_memory.get_bounce_performance()
                    consecutive = self.model_trainer.trade_memory.get_consecutive_performance()

                    # Add the same memory features as used in training
                    features.update({
                        'memory_win_rate': recent_perf['win_rate'],
                        'memory_avg_pnl': recent_perf['avg_pnl'],
                        'memory_total_trades': recent_perf['total_trades'],
                        'bounce_win_rate': bounce_perf['bounce_win_rate'],
                        'bounce_avg_pnl': bounce_perf['bounce_avg_pnl'],
                        'bounce_trade_count': bounce_perf['bounce_trades'],
                        'consecutive_wins': max(0, consecutive),
                        'consecutive_losses': max(0, -consecutive),
                        'market_volatility_regime': 0.5,  # Neutral regime for testing
                        'trend_strength': 0.0,  # Neutral trend for testing
                    })

                # Convert to DataFrame and ensure exact feature consistency with training
                feature_df = pd.DataFrame([features])

                # Critical: Use ONLY the features that were present during training
                missing_features = []
                for col in feature_columns:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0  # Default value for missing features
                        missing_features.append(col)

                # Remove any extra features not used in training
                feature_df = feature_df[feature_columns]

                # Log feature mismatches for debugging (only first occurrence)
                if i == 0 and missing_features:
                    print(f"   ⚠️  Missing {len(missing_features)} training features, using defaults")

                # Make prediction with probability analysis
                probabilities = trained_model.predict_proba(feature_df)[0]
                prediction_encoded = trained_model.predict(feature_df)[0]
                base_prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                confidence = probabilities.max()

                # Get class probabilities
                classes = label_encoder.classes_
                prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}

                # Apply MEMORY & BOUNCE-ENHANCED prediction logic
                buy_prob = prob_dict.get('buy', 0)
                sell_prob = prob_dict.get('sell', 0)
                hold_prob = prob_dict.get('hold', 0)

                # BOUNCE DETECTION & SIGNAL INJECTION
                current_price = float(row['close'])
                bounce_multiplier = 1.0
                bounce_info = ""

                # Check for bounce opportunities (if enabled)
                if self.model_trainer.enable_bounce_detection:
                    # Determine signal type for bounce detection
                    signal_type = 'BUY' if buy_prob > sell_prob else 'SELL'

                    # Detect bounce opportunity (convert ChartInterval keys to strings)
                    levels_dict = {str(k): v for k, v in self.trader.current_levels.items()}
                    is_bounce, bounce_strength, bounce_level = self.model_trainer.bounce_detector.detect_bounce_opportunity(
                        current_price, levels_dict, signal_type
                    )

                    if is_bounce and bounce_level:
                        # BOOST SIGNAL ON BOUNCE!
                        bounce_multiplier = 1.0 + (bounce_strength * 0.3)  # Up to 30% boost
                        bounce_info = f" | 🎯BOUNCE {bounce_level.level_type} @{bounce_level.price:.2f} (x{bounce_strength:.1f})"

                        # Apply bounce boost
                        if signal_type == 'BUY':
                            buy_prob = min(0.95, buy_prob * bounce_multiplier)
                        else:
                            sell_prob = min(0.95, sell_prob * bounce_multiplier)

                # MEMORY-ENHANCED THRESHOLDS
                memory_perf = self.model_trainer.trade_memory.get_recent_performance()
                memory_multiplier = 1.0

                # Adjust thresholds based on recent performance
                if memory_perf['win_rate'] > 0.6:  # Good recent performance
                    memory_multiplier = 0.9  # Lower thresholds (more aggressive)
                elif memory_perf['win_rate'] < 0.4:  # Poor recent performance
                    memory_multiplier = 1.1  # Higher thresholds (more conservative)

                # Apply memory-adjusted thresholds
                adj_buy_threshold = buy_threshold * memory_multiplier
                adj_sell_threshold = sell_threshold * memory_multiplier
                adj_aggressive_threshold = aggressive_threshold * memory_multiplier

                # Enhanced prediction logic with memory & bounce
                if buy_prob > adj_buy_threshold and buy_prob > sell_prob:
                    final_prediction = 'buy'
                    final_confidence = buy_prob
                elif sell_prob > adj_sell_threshold and sell_prob > buy_prob:
                    final_prediction = 'sell'
                    final_confidence = sell_prob
                elif buy_prob > adj_aggressive_threshold or sell_prob > adj_aggressive_threshold:
                    if buy_prob > sell_prob:
                        final_prediction = 'buy'
                        final_confidence = buy_prob
                    else:
                        final_prediction = 'sell'
                        final_confidence = sell_prob
                else:
                    final_prediction = 'hold'
                    final_confidence = hold_prob

                signals.append({
                    'datetime': row['datetime'],
                    'close': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'open': row['open'] if pd.notna(row['open']) else row['close'],
                    'volume': row['volume'],
                    'action': final_prediction,
                    'confidence': final_confidence,
                    'buy_prob': buy_prob,
                    'sell_prob': sell_prob,
                    'hold_prob': hold_prob,
                    'reasoning': f'{model_type} ({main_timeframe}): {final_prediction} ({final_confidence:.1%}) ' +
                                 f'[B:{buy_prob:.1%} S:{sell_prob:.1%} H:{hold_prob:.1%}]' + bounce_info
                })

            except Exception as e:
                print(f"   Error processing candle {i}: {e}")
                continue

        if not signals:
            print("❌ No predictions generated")
            return None

        signals_df = pd.DataFrame(signals)

        # Summary
        action_counts = signals_df['action'].value_counts()
        print(f"\n📊 PREDICTION RESULTS:")
        for action, count in action_counts.items():
            pct = count / len(signals_df) * 100
            print(f"   {action.upper()}: {count} ({pct:.1f}%)")

        avg_conf = signals_df['confidence'].mean()
        print(f"   Average Confidence: {avg_conf:.1%}")

        return signals_df
