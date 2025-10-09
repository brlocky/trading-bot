"""
Simple Model Trainer - Main training interface for the trading system
"""

import pandas as pd
import numpy as np
import os
import joblib
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb
import warnings

from trading.autonomous_trader import AutonomousTrader

warnings.filterwarnings('ignore')


class SimpleModelTrainer:
    """
    Simple and clean model trainer for trading signals.

    Uses AutonomousTrader as the single source of truth for feature calculation.
    """

    def __init__(self):
        self.model_data = {}
        self.is_trained = False
        self.end_time: Optional[pd.Timestamp] = None
        self.start_time: Optional[pd.Timestamp] = None

        # Default parameters
        self.profit_threshold = 2.0
        self.loss_threshold = -1.5
        self.lookforward_periods = [5, 10, 20]
        self.model_path = "src/models/simple_trading_model.joblib"
        self.start_time: Optional[pd.Timestamp] = None
        self.end_time: Optional[pd.Timestamp] = None

        # Initialize AutonomousTrader (SINGLE SOURCE OF TRUTH for features!)
        # Uses hardcoded middleware and level extraction configurations
        self.autonomous_trader = AutonomousTrader()

        # Use AutonomousTrader's memory and bounce systems (no duplicates!)
        self.trade_memory = self.autonomous_trader.trade_memory
        self.bounce_detector = self.autonomous_trader.bounce_detector
        self.enable_memory_features = True
        self.enable_bounce_detection = True

    def configure_training(self,
                           profit_threshold: float = 2.0,
                           loss_threshold: float = -1.5,
                           lookforward_periods: List[int] = [5, 10, 20],
                           start_time: Optional[pd.Timestamp] = None,
                           end_time: Optional[pd.Timestamp] = None):
        """
        Configure training parameters and optional time range filtering

        Args:
            profit_threshold: Minimum % gain to label as BUY signal
            loss_threshold: Maximum % loss to label as SELL signal
            lookforward_periods: List of forward-looking periods for labels
            start_time: Optional start timestamp to filter training data (inclusive)
            end_time: Optional end timestamp to filter training data (inclusive)
        """
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.lookforward_periods = lookforward_periods
        self.start_time = start_time
        self.end_time = end_time

    def _load_dataframes_from_files(self, level_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load all JSON files into DataFrames using centralized DataLoader.

        Args:
            level_files: Dict of timeframe -> file path

        Returns:
            Dict of timeframe -> DataFrame
        """
        from training.data_loader import DataLoader

        return DataLoader.load_multiple_json_files(
            level_files,
            exclude_keys=['parquet_path'],
            set_timestamp_index=True
        )

    def _prepare_training_data(
        self,
        main_df: pd.DataFrame,
        data_dfs: Dict[str, pd.DataFrame],
        timeframe: str = '15m'
    ) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:

        print("\n" + "=" * 70)
        print("🚀 PREPARING TRAINING DATA (OPTIMIZED)")
        print("=" * 70)
        start_time = datetime.now()

        try:
            # ============================================================
            # STEP 1: CALCULATE TA FEATURES FOR FULL DATASET (BEFORE FILTERING)
            # ============================================================
            # IMPORTANT: Calculate TA features on FULL historical data first
            # This ensures lagging indicators (200 SMA, 100 EMA, etc.) have proper history
            print("\n📈 Step 1/4: Calculating TA features for FULL dataset...")
            print("   Using {len(main_df):,} candles (includes all historical data)")
            ta_start = datetime.now()

            # Calculate TA features once for the FULL DataFrame (no time filter!)
            ta_features_full = self.autonomous_trader._calculate_ta_features(
                data=main_df,
                timeframe=timeframe,
                use_log_scale=True
            )

            ta_time = (datetime.now() - ta_start).total_seconds()
            print(f"   ✅ TA features calculated in {ta_time:.2f}s ({len(ta_features_full.columns)} features)")

            # ============================================================
            # STEP 2: FILTER DATA TO TRAINING RANGE
            # ============================================================
            print(f"\n📅 Step 2/4: Filtering to training time range...")

            # Now filter to the training time range
            df = main_df
            if self.start_time is not None:
                df = df[df.index >= self.start_time]
                print(f"   ⏰ Start time: {self.start_time}")

            if self.end_time is not None:
                df = df[df.index <= self.end_time]
                print(f"   ⏰ End time: {self.end_time}")

            if len(df) == 0:
                print(f"❌ No data in specified time range! {main_df.index[0]} to {main_df.index[-1]}")
                return None, None

            print(f"   ✅ Training range: {len(df)} candles ({df.index[0]} to {df.index[-1]})")

            # Filter TA features to match training range
            ta_features_df = ta_features_full.loc[df.index]
            print(f"   ✅ TA features filtered: {len(ta_features_df)} samples")

            # ============================================================
            # STEP 3: PRE-PROCESS LEVELS FOR TRAINING RANGE ONLY
            # ============================================================
            print(f"\n📊 Step 3/4: Pre-processing levels for {len(df)} candles...")
            preprocess_start = datetime.now()

            all_levels = []
            for i in range(len(df)):
                current_price = df.close.iloc[i]
                levels_raw = self.autonomous_trader.level_extractor.deserialize_levels_json(df['levels_json'].iloc[i])
                levels = self.autonomous_trader.level_extractor.convert_raw_to_levelinfo(levels_raw, float(current_price))
                all_levels.append(levels)

            preprocess_time = (datetime.now() - preprocess_start).total_seconds()
            print(f"   ✅ Levels pre-processed in {preprocess_time:.2f}s")

            # ============================================================
            # STEP 2: CALCULATE TA FEATURES ONCE FOR ENTIRE DATAFRAME
            # ============================================================
            print("\n Step 2/4: Calculating TA features for entire dataset...")
            ta_start = datetime.now()

            # Calculate TA features once for the full DataFrame
            ta_features_df = self.autonomous_trader._calculate_ta_features(
                data=df,
                timeframe=timeframe,
                use_log_scale=True
            )

            ta_time = (datetime.now() - ta_start).total_seconds()
            print(f"   ✅ TA features calculated in {ta_time:.2f}s ({len(ta_features_df.columns)} features)")

            # ============================================================
            # STEP 3: CALCULATE LEVEL FEATURES PER CANDLE (FAST!)
            # ============================================================
            print("\n🎯 Step 3/4: Calculating level features per candle...")
            level_start = datetime.now()

            from extraction.feature_engineer import LevelBasedFeatureEngineer
            level_engineer = LevelBasedFeatureEngineer()

            all_level_features = []
            for i in tqdm(range(len(df)), desc="Level features"):
                current_price = df.close.iloc[i]
                current_volume = df.volume.iloc[i]
                levels = all_levels[i]  # Pre-processed levels (no JSON parsing!)

                # Calculate level features for SINGLE candle (fast!)
                level_features_dict = level_engineer.create_level_features(
                    current_price=float(current_price),
                    current_volume=float(current_volume),
                    levels=levels
                )

                all_level_features.append(level_features_dict)

            # Convert list of dicts to DataFrame
            level_features_df = pd.DataFrame(all_level_features, index=df.index)

            level_time = (datetime.now() - level_start).total_seconds()
            print(f"   ✅ Level features calculated in {level_time:.2f}s ({len(level_features_df.columns)} features)")

            # ============================================================
            # STEP 4: ADD MEMORY FEATURES (ZEROS FOR TRAINING)
            # ============================================================
            print(f"\n💾 Step 4/4: Adding memory features...")
            memory_start = datetime.now()

            # Create 10 memory feature columns filled with zeros
            # (Memory features are for live trading alignment, not used in training)
            memory_feature_names = [
                'recent_trades', 'recent_wins', 'recent_losses',
                'avg_win_pct', 'avg_loss_pct', 'win_rate',
                'avg_holding_periods', 'total_profit_loss',
                'consecutive_wins', 'consecutive_losses'
            ]

            memory_features_df = pd.DataFrame(
                0.0,
                index=df.index,
                columns=memory_feature_names
            )

            memory_time = (datetime.now() - memory_start).total_seconds()
            print(f"   ✅ Memory features added in {memory_time:.3f}s ({len(memory_features_df.columns)} features)")

            # ============================================================
            # COMBINE ALL FEATURES
            # ============================================================
            print(f"\n🔗 Combining all feature groups...")
            combine_start = datetime.now()

            features_df = pd.concat([
                ta_features_df,
                level_features_df,
                memory_features_df
            ], axis=1)

            combine_time = (datetime.now() - combine_start).total_seconds()
            print(f"   ✅ Features combined in {combine_time:.3f}s")

            if features_df is None or len(features_df) == 0:
                print("❌ No training data prepared")
                return None, None

            # Feature breakdown:
            # - 45 TA features (from _calculate_ta_features)
            # - 52 Level features (from _calculate_level_features)
            # - 10 Memory features (zeros for alignment)
            # = 107 total features

            print(f"\n✅ Features ready: {len(features_df)} samples, {len(features_df.columns)} features")
            print(f"   TA: {len(ta_features_df.columns)}, Level: {len(level_features_df.columns)}, Memory: {len(memory_features_df.columns)}")

            # Generate labels based on future price movements + TA/level context
            print("🏷️  Generating trading labels with quality scores...")
            labels_dict = self._generate_labels_with_quality(
                df=df,
                ta_features=ta_features_df,
                level_features=level_features_df,
                all_levels=all_levels
            )

            # Align features with labels (features_df already has same index as df)
            features_df = features_df.loc[df.index]

            print(f"✅ Training data ready: {len(features_df)} samples, {len(features_df.columns)} features")

            # Clean NaN values
            total_nans = features_df.isnull().sum().sum()
            if total_nans > 0:
                print(f"🧹 Cleaning {total_nans} NaN values...")
                features_df = features_df.fillna(0)

            # Clean infinite values
            inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                print(f"🧹 Cleaning {inf_count} infinite values...")
                features_df = features_df.replace([np.inf, -np.inf], 0)

            # Convert labels (now with quality scores: 0=hold, 1=low quality, 2=high quality)
            labels = []
            for i in range(len(features_df)):
                buy_signal = labels_dict['buy'][i]
                sell_signal = labels_dict['sell'][i]

                if buy_signal == 2:
                    labels.append('buy_strong')  # High quality BUY
                elif buy_signal == 1:
                    labels.append('buy_weak')    # Low quality BUY
                elif sell_signal == 2:
                    labels.append('sell_strong')  # High quality SELL
                elif sell_signal == 1:
                    labels.append('sell_weak')   # Low quality SELL
                else:
                    labels.append('hold')

            # Check label distribution
            label_counts = pd.Series(labels).value_counts()
            print("📊 Label distribution:")
            for label, count in label_counts.items():
                percentage = count/len(labels)*100
                print(f"   {label}: {count} ({percentage:.1f}%)")

            # Calculate and display total time taken
            total_time = (datetime.now() - start_time).total_seconds()
            print("\n" + "=" * 70)
            print("✅ TRAINING DATA PREPARATION COMPLETE")
            print("=" * 70)
            print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"📊 Total samples: {len(features_df):,}")
            print(f"📊 Total features: {len(features_df.columns)}")
            print(f"⚡ Average time per sample: {total_time/len(features_df):.4f} seconds")
            print("=" * 70 + "\n")

            return features_df, labels

        except Exception as e:
            print(f"❌ Data preparation error: {e}")
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"⏱️  Failed after {total_time:.2f} seconds")
            return None, None

    def _generate_labels_with_quality(
        self,
        df: pd.DataFrame,
        ta_features: pd.DataFrame,
        level_features: pd.DataFrame,
        all_levels: List
    ) -> Dict[str, List[int]]:
        """
        Generate trading labels with quality scores based on:
        1. Future price movements (like old method)
        2. TA confluence (RSI, trend alignment)
        3. Level confluence (support/resistance proximity)
        4. Risk/Reward ratio (TP from volume profile, SL from pivots)

        Args:
            df: DataFrame with OHLCV data
            ta_features: TA features DataFrame
            level_features: Level features DataFrame
            all_levels: Pre-processed levels per candle

        Returns:
            Dict with 'buy', 'sell' signal lists (0=hold, 1=low quality, 2=high quality)
        """
        print("🏷️  Generating quality-scored labels...")

        buy_signals = []
        sell_signals = []

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Quality thresholds
        MIN_QUALITY = 0.4  # Minimum quality to label as buy/sell
        HIGH_QUALITY_THRESHOLD = 0.55  # Threshold for high-quality label (lowered from 0.7 - max observed quality is 0.533)
        MIN_RR_RATIO = 1.5  # Minimum risk/reward ratio

        for i in tqdm(range(len(df)), desc="Quality labels"):
            buy_signal = 0
            sell_signal = 0

            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # ============================================================
            # STEP 1: Check if price moves significantly (original logic)
            # ============================================================
            price_hit_profit = False
            price_hit_loss = False

            for period in self.lookforward_periods:
                if i + period < len(closes):
                    future_price = closes[i + period]
                    pct_change = ((future_price - current_price) / current_price) * 100

                    if pct_change >= self.profit_threshold:
                        price_hit_profit = True
                        break
                    elif pct_change <= self.loss_threshold:
                        price_hit_loss = True
                        break

            # Skip if no significant price movement
            if not price_hit_profit and not price_hit_loss:
                buy_signals.append(0)
                sell_signals.append(0)
                continue

            # ============================================================
            # STEP 2-5: Use BounceDetector for ALL quality calculations
            # ============================================================
            try:
                # Flatten all_levels dict to list
                all_level_list = []
                if isinstance(all_levels[i], dict):
                    for tf_levels in all_levels[i].values():
                        all_level_list.extend(tf_levels)
                else:
                    all_level_list = all_levels[i]

                signal_type = 'BUY' if price_hit_profit else 'SELL'

                # Get RSI for quality calculation
                rsi = ta_features['rsi'].iloc[i] if 'rsi' in ta_features.columns else 50.0

                # === SINGLE SOURCE OF TRUTH: BounceDetector ===
                # Calculates quality score including:
                # - Level proximity (30%)
                # - RSI confluence (20%)
                # - Level strength (20%)
                # - MACD alignment (optional, via TA features)
                # - BB position (optional, via TA features)
                base_quality = self.bounce_detector.calculate_labeling_quality(
                    current_price=current_price,
                    levels=all_level_list,
                    signal_type=signal_type,
                    rsi=rsi
                )

                # Add MACD confluence bonus (10% weight)
                macd_bonus = 0.0
                if 'macd' in ta_features.columns and 'macd_signal' in ta_features.columns:
                    macd = ta_features['macd'].iloc[i]
                    macd_signal = ta_features['macd_signal'].iloc[i]

                    if signal_type == 'BUY' and macd > macd_signal:
                        macd_bonus = 0.1
                    elif signal_type == 'SELL' and macd < macd_signal:
                        macd_bonus = 0.1

                # Add Bollinger Bands confluence bonus (10% weight)
                bb_bonus = 0.0
                if 'bb_position' in ta_features.columns:
                    bb_pos = ta_features['bb_position'].iloc[i]
                    if signal_type == 'BUY' and bb_pos < 0.3:
                        bb_bonus = 0.1
                    elif signal_type == 'SELL' and bb_pos > 0.7:
                        bb_bonus = 0.1

                # Calculate R/R ratio quality (30% weight)
                rr_quality = 0.0
                sl_price = self.bounce_detector.find_pivot_stop_loss(
                    current_price, all_level_list, signal_type
                )
                tp_price = self.bounce_detector.find_volume_profile_target(
                    current_price, all_level_list, signal_type
                )

                if sl_price and tp_price:
                    sl_dist = abs(current_price - sl_price)
                    tp_dist = abs(tp_price - current_price)

                    if sl_dist > 0:
                        rr_ratio = tp_dist / sl_dist

                        if rr_ratio >= 3.0:
                            rr_quality = 1.0
                        elif rr_ratio >= 2.0:
                            rr_quality = 0.8
                        elif rr_ratio >= MIN_RR_RATIO:
                            rr_quality = 0.6
                        else:
                            rr_quality = 0.3
                    else:
                        rr_quality = 0.5
                else:
                    rr_quality = 0.5

                # Final quality = Base (60%) + TA Bonuses (10%) + R/R (30%)
                total_quality = (base_quality * 0.6) + macd_bonus + bb_bonus + (rr_quality * 0.3)

            except Exception:
                print("❌ Quality calculation error, defaulting to 0.0")
                total_quality = 0.0  # Fallback if calculation fails

            # Generate final signal based on quality
            if price_hit_profit and total_quality >= MIN_QUALITY:
                if total_quality >= HIGH_QUALITY_THRESHOLD:
                    buy_signal = 2  # High quality BUY
                else:
                    buy_signal = 1  # Low quality BUY
            elif price_hit_loss and total_quality >= MIN_QUALITY:
                if total_quality >= HIGH_QUALITY_THRESHOLD:
                    sell_signal = 2  # High quality SELL
                else:
                    sell_signal = 1  # Low quality SELL

            buy_signals.append(buy_signal)
            sell_signals.append(sell_signal)

        # Count signal distribution
        buy_count = sum(1 for s in buy_signals if s > 0)
        buy_high_quality = sum(1 for s in buy_signals if s == 2)
        sell_count = sum(1 for s in sell_signals if s > 0)
        sell_high_quality = sum(1 for s in sell_signals if s == 2)
        hold_count = len(buy_signals) - buy_count - sell_count

        print(f"   BUY signals: {buy_count} ({buy_count/len(buy_signals)*100:.1f}%) | High quality: {buy_high_quality}")
        print(f"   SELL signals: {sell_count} ({sell_count/len(sell_signals)*100:.1f}%) | High quality: {sell_high_quality}")
        print(f"   HOLD signals: {hold_count} ({hold_count/len(buy_signals)*100:.1f}%)")

        return {'buy': buy_signals, 'sell': sell_signals}

    # Note: _add_memory_features() was removed - memory features now come from
    # AutonomousTrader.get_all_features() automatically

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _train_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train_encoded: np.ndarray, y_test_encoded: np.ndarray) -> Tuple[Any, str, Dict]:
        """Train model with GPU only - no CPU fallback"""
        gpu_available = self._detect_gpu()
        print(f"🖥️  GPU Detection: {'✅ Available' if gpu_available else '❌ Not Available'}")

        if not gpu_available:
            raise RuntimeError("❌ GPU not available! Training requires GPU. Please ensure NVIDIA GPU and drivers are properly installed.")

        results: Dict[str, Union[float, str, None]] = {'training_time': None, 'device': 'GPU'}

        print("   🚀 Training with GPU...")
        train_start_time = time.time()

        # OPTIMIZED THREADING: Using 8 threads for 8-core system
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, tree_method='gpu_hist', gpu_id=0,
            eval_metric='mlogloss', nthread=8
        )

        model.fit(X_train, y_train_encoded)
        training_time = time.time() - train_start_time

        accuracy = model.score(X_test, y_test_encoded)
        print(f"     ✅ GPU Training Complete: {training_time:.2f}s | Accuracy: {accuracy:.1%}")

        results['training_time'] = training_time
        model_type = "XGBoost-GPU"

        return model, model_type, results

    def train_model(self, training_files: Dict[str, str], level_timeframes: List[str] = ['M', 'W', 'D', '1h', '15m']) -> bool:
        """Train the model with the given files"""
        print("🎓 TRAINING MODEL")
        print("=" * 30)
        time_frame_to_train = '15m'

        # Validate all required files exist
        print("\n📋 Validating training files...")

        # Check main training file
        main_file = training_files.get(time_frame_to_train)
        if not main_file:
            raise ValueError(f"Main training timeframe '{time_frame_to_train}' not found in training_files")

        if not os.path.exists(main_file):
            raise FileNotFoundError(f"Main training file not found at {main_file}")

        print(f"   ✅ Main file ({time_frame_to_train}): {main_file}")

        # Check parquet file
        parquet_path = training_files.get('parquet_path')
        if not parquet_path:
            raise ValueError("'parquet_path' not found in training_files")

        print(f"   ✅ Parquet file: {parquet_path}")

        # Check all level timeframe files
        print(f"\n📊 Validating level timeframes: {level_timeframes}")
        missing_files = []

        for tf in level_timeframes:
            if tf not in training_files:
                missing_files.append(f"{tf} (not in training_files dict)")
                continue

            if not os.path.exists(training_files[tf]):
                missing_files.append(f"{tf} (file not found: {training_files[tf]})")

        # Report any missing files (after checking ALL timeframes)
        if missing_files:
            print("\n❌ Missing required files:")
            for missing in missing_files:
                print(f"   • {missing}")
            raise FileNotFoundError(f"Missing {len(missing_files)} required level timeframe file(s)")

        print(f"\n✅ All {len(training_files)} level timeframe files validated!")
        print(f"   Level timeframes: {list(training_files.keys())}")

        print("\n📦 Loading precomputed features from parquet...")
        main_df = self.autonomous_trader.level_extractor.load_precomputed_levels(parquet_path)
        print(f"   ✅ Loaded {len(main_df):,} candles from precomputed parquet")

        print("\n📂 Loading level timeframe data...")
        data_dfs = self._load_dataframes_from_files(training_files)

        # Prepare data
        features_df, labels = self._prepare_training_data(main_df, data_dfs, timeframe=time_frame_to_train)
        if features_df is None or labels is None:
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = np.asarray(label_encoder.fit_transform(y_train))
        y_test_encoded = np.asarray(label_encoder.transform(y_test))

        print(f"🔥 Training with {len(X_train)} samples, {len(X_train.columns)} features")

        # Train model
        trained_model, model_type, performance = self._train_model(X_train, X_test, y_train_encoded, y_test_encoded)

        # Final accuracy
        test_accuracy = trained_model.score(X_test, y_test_encoded)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': trained_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n✅ TRAINING COMPLETE!")
        print(f"   Model: {model_type}")
        print(f"   Test Accuracy: {test_accuracy:.1%}")
        print("\n🔍 Top 10 Features:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.3f}")

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.model_data = {
            'model': trained_model,
            'label_encoder': label_encoder,
            'feature_columns': list(X_train.columns),
            'label_classes': list(label_encoder.classes_),
            'accuracy': test_accuracy,
            'model_type': model_type,
            'training_performance': performance
        }

        joblib.dump(self.model_data, self.model_path)
        print(f"💾 Model saved to: {self.model_path}")

        self.is_trained = True
        return True

    def load_model(self) -> bool:
        """Load trained model"""
        if not os.path.exists(self.model_path):
            print("❌ No trained model found!")
            return False

        print(f"📁 Loading model from: {self.model_path}")
        self.model_data = joblib.load(self.model_path)

        model_type = self.model_data.get('model_type', 'Unknown')
        accuracy = self.model_data.get('accuracy', 0)

        print(f"✅ Model loaded: {model_type} (Accuracy: {accuracy:.1%})")

        self.is_trained = True
        return True

    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_trained:
            return {}

        return {
            'model_type': self.model_data.get('model_type', 'Unknown'),
            'accuracy': self.model_data.get('accuracy', 0),
            'features': len(self.model_data.get('feature_columns', [])),
            'classes': self.model_data.get('label_classes', [])
        }
