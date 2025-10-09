"""
VectorBT Backtesting Engine

Replaces the manual candle-by-candle simulation with VectorBT's
vectorized backtesting for high-performance strategy evaluation.
"""

import pandas as pd
import vectorbt as vbt
from typing import Dict, Optional, Any, cast
from pathlib import Path

from core.trading_types import ChartInterval
from ta.technical_analysis import Pivot


class VectorBTBacktester:
    """
    VectorBT-powered backtesting engine for ML trading strategies.

    Uses AutonomousTrader as the single source of truth for feature calculation,
    ensuring consistency between training, backtesting, and live trading.

    This class provides:
    - Vectorized signal generation from ML predictions
    - High-performance backtesting with realistic commissions/slippage
    - Comprehensive performance metrics
    - Multi-timeframe support with intelligent caching
    - Automatic day change detection for level features
    """

    def __init__(
        self,
        trainer,
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
        middleware_config: Optional[Dict] = None,
    ):
        """
        Initialize VectorBT backtester.

        Args:
            trainer: SimpleModelTrainer instance with trained model
            initial_cash: Starting capital for backtesting
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            middleware_config: Optional custom middleware configuration per timeframe
                             (passed to trainer's AutonomousTrader)
        """
        self.trainer = trainer
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.portfolio = None
        self.signals_df: Optional[pd.DataFrame] = None

        # No more FeatureManager! Use trainer's AutonomousTrader instead
        # All feature calculation happens in AutonomousTrader (single source of truth)

    def _load_levels_from_df(self, data_dfs: Dict[ChartInterval, pd.DataFrame],
                             last_pivot: Pivot, live_timeframe: ChartInterval,
                             use_log_scale: bool) -> Dict:
        """
        Load levels from DataFrames using trainer's AutonomousTrader configured extractor.
        Uses the level_extraction_config from trainer's AutonomousTrader initialization.

        Args:
            data_dfs: Dict of timeframe -> DataFrame with levels
            max_date: Optional datetime to filter data (only use candles <= max_date)
                     Critical for backtesting to prevent data leakage!
        """
        # Use trainer's AutonomousTrader extractor (already configured with level_extraction_config)
        try:
            all_levels = self.trainer.autonomous_trader.level_extractor.extract_levels_from_dataframes(
                data_dfs,
                last_pivot=last_pivot,
                live_timeframe=live_timeframe,
                use_log_scale=use_log_scale
            )
            return all_levels
        except Exception as e:
            print(f"   ⚠️  Error loading levels: {e}")
            # Return empty dict for each timeframe
            return {tf: [] for tf in data_dfs.keys()}

    def generate_signals_from_parquet(
        self,
        parquet_path: str,
        buy_threshold: float = 0.10,
        sell_threshold: float = 0.10,
        timeframe: ChartInterval = '15m',
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        ULTRA-FAST signal generation from precomputed parquet file.

        This is the fastest method as it reuses the exact same logic from
        SimpleModelTrainer._prepare_training_data but for prediction instead of training.

        Args:
            parquet_path: Path to parquet file with precomputed levels
            buy_threshold: Minimum probability for BUY signal
            sell_threshold: Minimum probability for SELL signal
            timeframe: Timeframe for feature calculation
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with signals and probabilities
        """
        from datetime import datetime
        from tqdm import tqdm
        import numpy as np

        if not self.trainer.is_trained:
            raise ValueError("Model is not trained. Train the model first.")

        print("🚀 ULTRA-FAST BACKTESTING: Using precomputed parquet file...")
        print("   Maximum speed with bulk processing!\n")
        start_time_calc = datetime.now()

        # Load precomputed data from parquet
        print("📦 Loading precomputed features from parquet...")
        data = self.trainer.autonomous_trader.level_extractor.load_precomputed_levels(parquet_path)
        print(f"   ✅ Loaded {len(data):,} candles from parquet")

        # Apply time filtering if specified
        if start_time is not None:
            data = data[data.index >= start_time]
            print(f"   ⏰ Filtered from start_time: {len(data):,} candles remaining")

        if end_time is not None:
            data = data[data.index <= end_time]
            print(f"   ⏰ Filtered to end_time: {len(data):,} candles remaining")

        if len(data) == 0:
            raise ValueError("No data remaining after time filtering!")

        # Get model components
        model_data = self.trainer.model_data
        trained_model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_columns = model_data['feature_columns']

        # ============================================================
        # REUSE EXACT LOGIC FROM TRAINER._PREPARE_TRAINING_DATA
        # ============================================================

        # STEP 1: Calculate TA features for full dataset
        print("📈 Step 1/4: Calculating TA features for full dataset...")
        ta_start = datetime.now()

        ta_features_df = self.trainer.autonomous_trader._calculate_ta_features(
            data=data,
            timeframe=timeframe,
            use_log_scale=True
        )

        ta_time = (datetime.now() - ta_start).total_seconds()
        print(f"   ✅ TA features calculated in {ta_time:.2f}s ({len(ta_features_df.columns)} features)")

        # STEP 2: Pre-process levels for all candles (optimized)
        print(f"📊 Step 2/4: Pre-processing levels for {len(data)} candles...")
        preprocess_start = datetime.now()

        all_levels = []
        for i in range(len(data)):
            current_price = data.close.iloc[i]
            levels_raw = self.trainer.autonomous_trader.level_extractor.deserialize_levels_json(data['levels_json'].iloc[i])
            levels = self.trainer.autonomous_trader.level_extractor.convert_raw_to_levelinfo(levels_raw, float(current_price))
            all_levels.append(levels)

        preprocess_time = (datetime.now() - preprocess_start).total_seconds()
        print(f"   ✅ Levels pre-processed in {preprocess_time:.2f}s")

        # STEP 3: Calculate level features per candle (bulk optimized)
        print("🎯 Step 3/4: Calculating level features per candle...")
        level_start = datetime.now()

        from extraction.feature_engineer import LevelBasedFeatureEngineer
        level_engineer = LevelBasedFeatureEngineer()

        all_level_features = []
        for i in tqdm(range(len(data)), desc="Level features"):
            current_price = data.close.iloc[i]
            current_volume = data.volume.iloc[i]
            levels = all_levels[i]  # Pre-processed levels (no JSON parsing per iteration!)

            # Calculate level features for SINGLE candle (fast!)
            level_features_dict = level_engineer.create_level_features(
                current_price=float(current_price),
                current_volume=float(current_volume),
                levels=levels
            )
            all_level_features.append(level_features_dict)

        # Convert list of dicts to DataFrame
        level_features_df = pd.DataFrame(all_level_features, index=data.index)

        level_time = (datetime.now() - level_start).total_seconds()
        print(f"   ✅ Level features calculated in {level_time:.2f}s ({len(level_features_df.columns)} features)")

        # STEP 4: Add memory features (zeros for backtesting alignment)
        print("💾 Step 4/4: Adding memory features...")
        memory_start = datetime.now()

        memory_feature_names = [
            'recent_trades', 'recent_wins', 'recent_losses',
            'avg_win_pct', 'avg_loss_pct', 'win_rate',
            'avg_holding_periods', 'total_profit_loss',
            'consecutive_wins', 'consecutive_losses'
        ]

        memory_features_df = pd.DataFrame(
            0.0,
            index=data.index,
            columns=memory_feature_names
        )

        memory_time = (datetime.now() - memory_start).total_seconds()
        print(f"   ✅ Memory features added in {memory_time:.3f}s ({len(memory_features_df.columns)} features)")

        # ============================================================
        # COMBINE ALL FEATURES (SAME AS TRAINING)
        # ============================================================
        print("🔗 Combining all feature groups...")
        combine_start = datetime.now()

        features_df = pd.concat([
            ta_features_df,
            level_features_df,
            memory_features_df
        ], axis=1)

        combine_time = (datetime.now() - combine_start).total_seconds()
        print(f"   ✅ Features combined in {combine_time:.3f}s")

        print(f"✅ Features ready: {len(features_df)} samples, {len(features_df.columns)} features")
        print(f"   TA: {len(ta_features_df.columns)}, Level: {len(level_features_df.columns)}, Memory: {len(memory_features_df.columns)}")

        # Ensure exact feature match with training features
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Align features with training order
        features_aligned = features_df[feature_columns]

        # Clean NaN and infinite values
        features_aligned = features_aligned.fillna(0.0)
        features_aligned = features_aligned.replace([np.inf, -np.inf], 0.0)

        # ============================================================
        # BULK PREDICTION (VECTORIZED)
        # ============================================================
        print(f"🔮 Generating predictions for {len(features_aligned)} candles...")
        pred_start = datetime.now()

        # Generate predictions for ALL candles at once (vectorized)
        probabilities = trained_model.predict_proba(features_aligned)

        pred_time = (datetime.now() - pred_start).total_seconds()
        print(f"   ✅ Predictions generated in {pred_time:.2f}s")

        # ============================================================
        # CREATE SIGNALS DATAFRAME
        # ============================================================
        signals_df = pd.DataFrame(index=data.index)
        signals_df['close'] = data['close']

        # Add probabilities for each class
        classes = label_encoder.classes_
        for j, class_name in enumerate(classes):
            signals_df[f'{class_name}_prob'] = probabilities[:, j]

        # Support both 3-class and 5-class models
        if 'buy_prob' not in signals_df.columns:
            # 5-class model: aggregate strong + weak probabilities
            buy_strong = signals_df.get('buy_strong_prob', pd.Series(0.0, index=signals_df.index))
            buy_weak = signals_df.get('buy_weak_prob', pd.Series(0.0, index=signals_df.index))
            sell_strong = signals_df.get('sell_strong_prob', pd.Series(0.0, index=signals_df.index))
            sell_weak = signals_df.get('sell_weak_prob', pd.Series(0.0, index=signals_df.index))

            buy_prob = buy_strong + buy_weak
            sell_prob = sell_strong + sell_weak

            signals_df['buy_prob'] = buy_prob
            signals_df['sell_prob'] = sell_prob

            print(f"\n🔍 5-Class Model Detected - Aggregating probabilities:")
            print(f"   BUY = buy_strong + buy_weak")
            print(f"   SELL = sell_strong + sell_weak")
            print(f"   Max BUY prob: {buy_prob.max():.4f}")
            print(f"   Max SELL prob: {sell_prob.max():.4f}")
        else:
            # 3-class model: use existing probabilities
            buy_prob = signals_df['buy_prob']
            sell_prob = signals_df['sell_prob']

        # Determine signals based on thresholds
        hold_prob = signals_df.get('hold_prob', pd.Series(0.0, index=signals_df.index))

        # Apply thresholds
        signals_df['signal'] = 'HOLD'
        signals_df['confidence'] = hold_prob

        buy_mask = (buy_prob > buy_threshold) & (buy_prob > sell_prob)
        signals_df.loc[buy_mask, 'signal'] = 'BUY'
        signals_df.loc[buy_mask, 'confidence'] = buy_prob[buy_mask]

        sell_mask = (sell_prob > sell_threshold) & (sell_prob > buy_prob)
        signals_df.loc[sell_mask, 'signal'] = 'SELL'
        signals_df.loc[sell_mask, 'confidence'] = sell_prob[sell_mask]

        # Create entry/exit signals for VectorBT
        buy_signals = (signals_df['signal'] == 'BUY').sum()
        sell_signals = (signals_df['signal'] == 'SELL').sum()

        print("\n🔍 Signal Analysis:")
        print(f"   BUY signals: {buy_signals}")
        print(f"   SELL signals: {sell_signals}")

        if buy_signals == 0 and sell_signals > 0:
            print("⚠️  No BUY signals detected - adjusting strategy...")
            buy_threshold_auto = buy_prob.quantile(0.9)  # Top 10% of BUY probabilities
            print(f"   Auto-adjusted BUY threshold: {buy_threshold_auto:.4f}")

            signals_df['entries'] = (buy_prob >= buy_threshold_auto).astype(int)
            signals_df['exits'] = (signals_df['signal'] == 'SELL').astype(int)
            signals_df.loc[buy_prob >= buy_threshold_auto, 'signal'] = 'BUY'

            new_buy_signals = signals_df['entries'].sum()
            new_sell_signals = signals_df['exits'].sum()
            print(f"   Adjusted BUY signals: {new_buy_signals}")
            print(f"   Adjusted SELL signals: {new_sell_signals}")
        else:
            signals_df['entries'] = (signals_df['signal'] == 'BUY').astype(int)
            signals_df['exits'] = (signals_df['signal'] == 'SELL').astype(int)

        # Calculate and display total time taken
        total_time = (datetime.now() - start_time_calc).total_seconds()
        print("\n" + "=" * 70)
        print("✅ ULTRA-FAST BACKTESTING COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"📊 Processed: {len(signals_df):,} candles")
        print(f"   BUY: {(signals_df['signal'] == 'BUY').sum()}")
        print(f"   SELL: {(signals_df['signal'] == 'SELL').sum()}")
        print(f"   HOLD: {(signals_df['signal'] == 'HOLD').sum()}")
        print(f"⚡ Average time per candle: {total_time/len(signals_df)*1000:.2f} milliseconds")
        print("=" * 70 + "\n")

        self.signals_df = signals_df
        return signals_df

    def generate_signals(
        self,
        data: pd.DataFrame,
        data_dfs: Dict[ChartInterval, pd.DataFrame],
        buy_threshold: float = 0.10,
        sell_threshold: float = 0.10,
        timeframe: ChartInterval = '15m',
    ) -> pd.DataFrame:
        """
        Generate trading signals with optional optimized bulk processing.

        Args:
            data: DataFrame with OHLCV data(must have datetime index)
            data_dfs: Dict of timeframe -> DataFrame with levels
            buy_threshold: Minimum probability for BUY signal
            sell_threshold: Minimum probability for SELL signal
            timeframe: Timeframe for feature calculation(e.g., '15m', '1h', 'D')
            use_optimized: If True, uses optimized bulk processing (recommended)
                          If False, uses progressive candle-by-candle (slower but transparent)

        Returns:
            DataFrame with columns: datetime, close, buy_prob, sell_prob, hold_prob,
                                   signal, confidence, entries, exits
        """
        if not self.trainer.is_trained:
            raise ValueError("Model is not trained. Train the model first.")

        return self._generate_signals_optimized(data, data_dfs, buy_threshold, sell_threshold, timeframe)

    def _generate_signals_optimized(
        self,
        data: pd.DataFrame,
        data_dfs: Dict[ChartInterval, pd.DataFrame],
        buy_threshold: float,
        sell_threshold: float,
        timeframe: ChartInterval,
    ) -> pd.DataFrame:
        """
        OPTIMIZED signal generation using bulk processing (based on trainer's _prepare_training_data).

        This method reuses the efficient bulk processing approach from SimpleModelTrainer
        for much faster backtesting. While we still need to extract levels per candle
        to prevent data leakage, we optimize by:
        1. Bulk TA feature calculation for entire dataset
        2. Bulk level feature processing (after level extraction)
        3. Vectorized predictions for all candles at once

        Key difference from training: No parquet file, so levels must be extracted
        from live timeframe data while preventing data leakage.
        """
        from datetime import datetime
        from tqdm import tqdm
        import numpy as np

        print(f"🚀 OPTIMIZED BACKTESTING: Processing {len(data)} candles...")
        print("   Using bulk processing for maximum speed!\n")
        start_time = datetime.now()

        # Get model components
        model_data = self.trainer.model_data
        trained_model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_columns = model_data['feature_columns']

        # ============================================================
        # STEP 1: CALCULATE TA FEATURES FOR FULL DATASET (OPTIMIZED)
        # ============================================================
        print("📈 Step 1/3: Calculating TA features for full dataset...")
        ta_start = datetime.now()

        # Calculate TA features once for the FULL DataFrame (optimized approach)
        ta_features_df = self.trainer.autonomous_trader._calculate_ta_features(
            data=data,
            timeframe=timeframe,
            use_log_scale=True
        )

        ta_time = (datetime.now() - ta_start).total_seconds()
        print(f"   ✅ TA features calculated in {ta_time:.2f}s ({len(ta_features_df.columns)} features)")

        # ============================================================
        # STEP 2: EXTRACT LEVELS FOR ALL CANDLES (WITH DATA LEAKAGE PREVENTION)
        # ============================================================
        print(f"📊 Step 2/3: Extracting levels for {len(data)} candles...")
        preprocess_start = datetime.now()

        # Extract levels from timeframe data for each candle
        # We must do this per candle to prevent data leakage in backtesting
        all_levels = []
        for i in tqdm(range(len(data)), desc="Extracting levels"):
            current_timestamp = cast(pd.Timestamp, data.index[i])
            current_price = data.close[i]

            # CRITICAL: Get only data up to current timestamp to prevent data leakage
            filtered_data_dfs = {}
            for tf, df in data_dfs.items():
                filtered_data_dfs[tf] = df[df.index <= current_timestamp]

            raw_levels = self._load_levels_from_df(
                filtered_data_dfs,
                last_pivot=(current_timestamp, current_price, None, None),
                live_timeframe=timeframe,
                use_log_scale=True
            )
            levels = self.trainer.autonomous_trader.level_extractor.convert_raw_to_levelinfo(
                raw_levels, current_price
            )
            all_levels.append(levels)

        preprocess_time = (datetime.now() - preprocess_start).total_seconds()
        print(f"   ✅ Levels extracted in {preprocess_time:.2f}s")

        # ============================================================
        # STEP 3: CALCULATE LEVEL FEATURES (BULK OPTIMIZED)
        # ============================================================
        print("🎯 Step 3/3: Calculating level features per candle...")
        level_start = datetime.now()

        from extraction.feature_engineer import LevelBasedFeatureEngineer
        level_engineer = LevelBasedFeatureEngineer()

        all_level_features = []
        for i in tqdm(range(len(data)), desc="Level features"):
            current_price = data.close.iloc[i]
            current_volume = data.volume.iloc[i]
            levels = all_levels[i]  # Pre-processed levels (no JSON parsing!)

            # Calculate level features for SINGLE candle (fast!)
            level_features_dict = level_engineer.create_level_features(
                current_price=float(current_price),
                current_volume=float(current_volume),
                levels=levels
            )
            all_level_features.append(level_features_dict)

        # Convert list of dicts to DataFrame
        level_features_df = pd.DataFrame(all_level_features, index=data.index)

        level_time = (datetime.now() - level_start).total_seconds()
        print(f"   ✅ Level features calculated in {level_time:.2f}s ({len(level_features_df.columns)} features)")

        # ============================================================
        # STEP 4: ADD MEMORY FEATURES (ZEROS FOR BACKTESTING)
        # ============================================================
        memory_feature_names = [
            'recent_trades', 'recent_wins', 'recent_losses',
            'avg_win_pct', 'avg_loss_pct', 'win_rate',
            'avg_holding_periods', 'total_profit_loss',
            'consecutive_wins', 'consecutive_losses'
        ]

        memory_features_df = pd.DataFrame(
            0.0,
            index=data.index,
            columns=memory_feature_names
        )

        # ============================================================
        # STEP 5: COMBINE ALL FEATURES
        # ============================================================
        features_df = pd.concat([
            ta_features_df,
            level_features_df,
            memory_features_df
        ], axis=1)

        # Ensure exact feature match with training features
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Align features with training order
        features_aligned = features_df[feature_columns]

        # Clean NaN and infinite values
        features_aligned = features_aligned.fillna(0.0)
        features_aligned = features_aligned.replace([np.inf, -np.inf], 0.0)

        # ============================================================
        # STEP 6: BULK PREDICTION (VECTORIZED)
        # ============================================================
        print(f"🔮 Generating predictions for {len(features_aligned)} candles...")
        pred_start = datetime.now()

        # Generate predictions for ALL candles at once (vectorized)
        probabilities = trained_model.predict_proba(features_aligned)

        pred_time = (datetime.now() - pred_start).total_seconds()
        print(f"   ✅ Predictions generated in {pred_time:.2f}s")

        # ============================================================
        # STEP 7: CREATE SIGNALS DATAFRAME
        # ============================================================
        signals_df = pd.DataFrame(index=data.index)
        signals_df['close'] = data['close']

        # Add probabilities for each class
        classes = label_encoder.classes_
        for j, class_name in enumerate(classes):
            signals_df[f'{class_name}_prob'] = probabilities[:, j]

        # Support both 3-class and 5-class models
        if 'buy_prob' not in signals_df.columns:
            # 5-class model: aggregate strong + weak probabilities
            buy_strong = signals_df.get('buy_strong_prob', pd.Series(0.0, index=signals_df.index))
            buy_weak = signals_df.get('buy_weak_prob', pd.Series(0.0, index=signals_df.index))
            sell_strong = signals_df.get('sell_strong_prob', pd.Series(0.0, index=signals_df.index))
            sell_weak = signals_df.get('sell_weak_prob', pd.Series(0.0, index=signals_df.index))

            buy_prob = buy_strong + buy_weak
            sell_prob = sell_strong + sell_weak

            signals_df['buy_prob'] = buy_prob
            signals_df['sell_prob'] = sell_prob

            print("\n🔍 5-Class Model Detected - Aggregating probabilities:")
            print(f"   Max BUY prob: {buy_prob.max():.4f}")
            print(f"   Max SELL prob: {sell_prob.max():.4f}")
        else:
            # 3-class model: use existing probabilities
            buy_prob = signals_df['buy_prob']
            sell_prob = signals_df['sell_prob']

        # Determine signals based on thresholds
        hold_prob = signals_df.get('hold_prob', pd.Series(0.0, index=signals_df.index))

        # Apply thresholds
        signals_df['signal'] = 'HOLD'
        signals_df['confidence'] = hold_prob

        buy_mask = (buy_prob > buy_threshold) & (buy_prob > sell_prob)
        signals_df.loc[buy_mask, 'signal'] = 'BUY'
        signals_df.loc[buy_mask, 'confidence'] = buy_prob[buy_mask]

        sell_mask = (sell_prob > sell_threshold) & (sell_prob > buy_prob)
        signals_df.loc[sell_mask, 'signal'] = 'SELL'
        signals_df.loc[sell_mask, 'confidence'] = sell_prob[sell_mask]

        # Create entry/exit signals for VectorBT
        buy_signals = (signals_df['signal'] == 'BUY').sum()
        sell_signals = (signals_df['signal'] == 'SELL').sum()

        print("\n🔍 Signal Analysis:")
        print(f"   BUY signals: {buy_signals}")
        print(f"   SELL signals: {sell_signals}")

        if buy_signals == 0 and sell_signals > 0:
            print("⚠️  No BUY signals detected - adjusting strategy...")
            buy_threshold_auto = buy_prob.quantile(0.9)  # Top 10% of BUY probabilities
            print(f"   Auto-adjusted BUY threshold: {buy_threshold_auto:.4f}")

            signals_df['entries'] = (buy_prob >= buy_threshold_auto).astype(int)
            signals_df['exits'] = (signals_df['signal'] == 'SELL').astype(int)
            signals_df.loc[buy_prob >= buy_threshold_auto, 'signal'] = 'BUY'

            new_buy_signals = signals_df['entries'].sum()
            new_sell_signals = signals_df['exits'].sum()
            print(f"   Adjusted BUY signals: {new_buy_signals}")
            print(f"   Adjusted SELL signals: {new_sell_signals}")
        else:
            signals_df['entries'] = (signals_df['signal'] == 'BUY').astype(int)
            signals_df['exits'] = (signals_df['signal'] == 'SELL').astype(int)

        # Calculate and display total time taken
        total_time = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 70)
        print("✅ OPTIMIZED BACKTESTING COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"📊 Processed: {len(signals_df):,} candles")
        print(f"   BUY: {(signals_df['signal'] == 'BUY').sum()}")
        print(f"   SELL: {(signals_df['signal'] == 'SELL').sum()}")
        print(f"   HOLD: {(signals_df['signal'] == 'HOLD').sum()}")
        print(f"⚡ Average time per candle: {total_time/len(signals_df)*1000:.2f} milliseconds")
        print("=" * 70 + "\n")

        self.signals_df = signals_df
        return signals_df

    def run_backtest(
        self,
        signals_df: Optional[pd.DataFrame] = None,
        freq: str = '15T',  # 15-minute frequency
    ) -> vbt.Portfolio:
        """
        Run VectorBT backtest on generated signals.

        Args:
            signals_df: DataFrame with entries/exits(uses self.signals_df if None)
            freq: Frequency string for VectorBT(e.g., '15T', '1H', '1D')

        Returns:
            VectorBT Portfolio object with results
        """
        if signals_df is None:
            if self.signals_df is None:
                raise ValueError("No signals available. Run generate_signals() first.")
            signals_df = self.signals_df

        print("\n🚀 Running VectorBT backtest...")
        print(f"   Initial Capital: ${self.initial_cash:,.2f}")
        print(f"   Commission: {self.commission:.2%}")
        print(f"   Slippage: {self.slippage:.2%}")

        # Create Portfolio using from_signals
        portfolio = vbt.Portfolio.from_signals(
            close=signals_df['close'],
            entries=signals_df['entries'],
            exits=signals_df['exits'],
            init_cash=self.initial_cash,
            fees=self.commission,
            slippage=self.slippage,
            freq=freq,
        )

        self.portfolio = portfolio
        print("✅ Backtest complete!")

        return portfolio

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        stats = self.portfolio.stats()

        # Convert to dictionary for easier access
        stats_dict = {
            'total_return': stats['Total Return [%]'],
            'annualized_return': stats.get('Annual Return [%]', 0),
            'sharpe_ratio': stats.get('Sharpe Ratio', 0),
            'max_drawdown': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'total_trades': stats['Total Trades'],
            'profit_factor': stats.get('Profit Factor', 0),
            'final_value': stats['End Value'],
        }

        return stats_dict

    def print_performance_summary(self):
        """Print a formatted performance summary."""
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        print("\n" + "=" * 60)
        print("📊 VECTORBT BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)

        stats = self.get_performance_stats()

        print("\n💰 Returns:")
        print(f"   Total Return:      {stats['total_return']:>10.2f}%")
        print(f"   Annualized Return: {stats['annualized_return']:>10.2f}%")
        print(f"   Final Value:       ${stats['final_value']:>10,.2f}")

        print("\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:      {stats['sharpe_ratio']:>10.2f}")
        print(f"   Max Drawdown:      {stats['max_drawdown']:>10.2f}%")

        print("\n🎯 Trading Performance:")
        print(f"   Total Trades:      {stats['total_trades']:>10}")
        print(f"   Win Rate:          {stats['win_rate']:>10.2f}%")
        print(f"   Profit Factor:     {stats['profit_factor']:>10.2f}")

        print("\n" + "=" * 60)

    def plot_results(self, use_widgets=False):
        """
        Generate interactive VectorBT plots.

        This creates:
        - Equity curve
        - Drawdown chart
        - Trade markers on price chart

        Args:
            use_widgets: If True, uses FigureWidget(requires anywidget).
                        If False, uses static Figure(no dependencies).
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        print("\n📊 Generating interactive plots...")

        try:
            # Try with widgets first
            if use_widgets:
                fig = self.portfolio.plot()
            else:
                # Use static plots (no widgets required)
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # Get portfolio value over time
                portfolio_value = self.portfolio.value()

                # Create subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Portfolio Value', 'Drawdown %')
                )

                # Add portfolio value trace
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value.values,
                        name='Portfolio Value',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )

                # Add initial cash line
                fig.add_hline(
                    y=self.initial_cash,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Initial Cash",
                    row=1, col=1
                )

                # Calculate and add drawdown
                drawdown = self.portfolio.drawdown()
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values * 100,  # Convert to percentage
                        name='Drawdown %',
                        fill='tozeroy',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )

                # Update layout
                fig.update_layout(
                    title='VectorBT Backtest Results',
                    height=800,
                    showlegend=True,
                    hovermode='x unified'
                )

                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Value ($)", row=1, col=1)
                fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

            fig.show()
            print("✅ Plots displayed!")

        except ImportError as e:
            if "anywidget" in str(e):
                print("\n⚠️  anywidget not installed!")
                print("   Solution 1: Restart the kernel (Kernel → Restart)")
                print("   Solution 2: Run: backtester.plot_results(use_widgets=False)")
                print("   Solution 3: Install anywidget and restart kernel")
            else:
                raise

    def get_trade_analysis(self) -> pd.DataFrame:
        """
        Get detailed trade-by-trade analysis.

        Returns:
            DataFrame with individual trade details
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        trades = self.portfolio.trades.records_readable
        return trades

    def export_results(self, output_dir: str = "backtest_results"):
        """
        Export backtest results to files.

        Args:
            output_dir: Directory to save results
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export signals
        signals_file = output_path / "signals.csv"
        self.signals_df.to_csv(signals_file)
        print(f"✅ Saved signals to: {signals_file}")

        # Export trades
        trades_file = output_path / "trades.csv"
        trades = self.get_trade_analysis()
        trades.to_csv(trades_file, index=False)
        print(f"✅ Saved trades to: {trades_file}")

        # Export performance stats
        stats_file = output_path / "performance_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(str(self.portfolio.stats()))
        print(f"✅ Saved performance stats to: {stats_file}")

        # Export equity curve
        equity_file = output_path / "equity_curve.csv"
        equity = self.portfolio.value()
        equity.to_csv(equity_file)
        print(f"✅ Saved equity curve to: {equity_file}")

        print(f"\n📁 All results exported to: {output_path.absolute()}")
