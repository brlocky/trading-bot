"""
Simple Model Trainer - Main training interface for the trading system
"""

import pandas as pd
import numpy as np
import os
import joblib
import time
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

from src.training.autonomous_trainer import AutonomousTraderTrainer
from src.memory.trade_memory import TradeMemoryManager
from src.detection.bounce_detector import BounceDetector

warnings.filterwarnings('ignore')


class SimpleModelTrainer:
    """
    Simple and clean model trainer for trading signals
    """

    def __init__(self):
        self.trainer = AutonomousTraderTrainer()
        self.model_data = {}
        self.is_trained = False

        # Default parameters
        self.profit_threshold = 2.0
        self.loss_threshold = -1.5
        self.lookforward_periods = [5, 10, 20]
        self.model_path = 'src/models/simple_trading_model.joblib'

        # Memory and bounce detection systems
        self.trade_memory = TradeMemoryManager()
        self.bounce_detector = BounceDetector()
        self.enable_memory_features = True
        self.enable_bounce_detection = True

    def configure_training(self, profit_threshold: float = 2.0,
                           loss_threshold: float = -1.5,
                           lookforward_periods: List[int] = [5, 10, 20]):
        """Configure training parameters"""
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.lookforward_periods = lookforward_periods

        # Update trainer
        self.trainer.profit_threshold = profit_threshold
        self.trainer.loss_threshold = loss_threshold
        self.trainer.lookforward_periods = lookforward_periods

    def _prepare_training_data(self, main_file: str, level_files: Dict[str, str]) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Prepare and clean training data"""
        print("📊 Preparing training data (optimized processing)...")

        try:
            # Get raw features and labels with parallel processing
            features_df, labels_dict = self.trainer.prepare_training_data(main_file, level_files)

            if features_df is None or len(features_df) == 0:
                print("❌ No training data prepared")
                return None, None

            print(f"✅ Training data ready: {len(features_df)} samples, {len(features_df.columns)} features")

            # ADD MEMORY-ENHANCED FEATURES
            if self.enable_memory_features:
                features_df = self._add_memory_features(features_df)
                print(f"🧠 Memory features added: {len(features_df.columns)} total features")

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

            # Convert labels
            labels = []
            for i in range(len(features_df)):
                buy_signal = labels_dict['buy'][i]
                sell_signal = labels_dict['sell'][i]

                if buy_signal == 1:
                    labels.append('buy')
                elif sell_signal == 1:
                    labels.append('sell')
                else:
                    labels.append('hold')

            # Check label distribution
            label_counts = pd.Series(labels).value_counts()
            print(f"📊 Label distribution:")
            for label, count in label_counts.items():
                percentage = count/len(labels)*100
                print(f"   {label}: {count} ({percentage:.1f}%)")

            return features_df, labels

        except Exception as e:
            print(f"❌ Data preparation error: {e}")
            return None, None

    def _add_memory_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        ADD MEMORY-ENHANCED FEATURES
        Add trade memory and market context features to improve predictions
        """
        print("🧠 Adding memory-enhanced features...")

        # Get current trade memory stats
        recent_perf = self.trade_memory.get_recent_performance()
        bounce_perf = self.trade_memory.get_bounce_performance()
        consecutive = self.trade_memory.get_consecutive_performance()

        num_samples = len(features_df)

        # Memory features (same for all samples during training)
        memory_features = pd.DataFrame({
            # Recent performance memory
            'memory_win_rate': [recent_perf['win_rate']] * num_samples,
            'memory_avg_pnl': [recent_perf['avg_pnl']] * num_samples,
            'memory_total_trades': [recent_perf['total_trades']] * num_samples,

            # Bounce-specific memory
            'bounce_win_rate': [bounce_perf['bounce_win_rate']] * num_samples,
            'bounce_avg_pnl': [bounce_perf['bounce_avg_pnl']] * num_samples,
            'bounce_trade_count': [bounce_perf['bounce_trades']] * num_samples,

            # Consecutive performance
            'consecutive_wins': [max(0, consecutive)] * num_samples,
            'consecutive_losses': [max(0, -consecutive)] * num_samples,

            # Market context features (simulated for training, real for testing)
            'market_volatility_regime': np.random.uniform(0.1, 2.0, num_samples),
            'trend_strength': np.random.uniform(-1.0, 1.0, num_samples),
        })

        # Combine with existing features
        enhanced_features = pd.concat([features_df, memory_features], axis=1)

        print(f"   ✅ Added {len(memory_features.columns)} memory features")
        print(f"   📊 Win Rate: {recent_perf['win_rate']:.1%} | Avg PnL: {recent_perf['avg_pnl']:.2f}%")
        print(f"   🎯 Bounce Win Rate: {bounce_perf['bounce_win_rate']:.1%}")
        print(f"   🔥 Consecutive: {consecutive} {'wins' if consecutive > 0 else 'losses' if consecutive < 0 else 'neutral'}")

        return enhanced_features

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

        results = {'training_time': None, 'device': 'GPU'}

        print("   🚀 Training with GPU...")
        start_time = time.time()

        # OPTIMIZED THREADING: Using 8 threads for 8-core system
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, tree_method='gpu_hist', gpu_id=0,
            eval_metric='mlogloss', nthread=8
        )

        model.fit(X_train, y_train_encoded)
        training_time = time.time() - start_time

        accuracy = model.score(X_test, y_test_encoded)
        print(f"     ✅ GPU Training Complete: {training_time:.2f}s | Accuracy: {accuracy:.1%}")

        results['training_time'] = training_time
        model_type = "XGBoost-GPU"

        return model, model_type, results

    def train_model(self, training_files: Dict[str, str], level_timeframes: List[str] = ['M', 'W', 'D', '1h']) -> bool:
        """Train the model with the given files"""
        print("🎓 TRAINING MODEL")
        print("=" * 30)

        # Prepare level files
        level_files = {tf: training_files[tf] for tf in level_timeframes if tf in training_files}
        main_file = training_files['15m']  # Use 15m as main timeframe

        print(f"Main timeframe: 15m")
        print(f"Level timeframes: {list(level_files.keys())}")

        # Prepare data
        features_df, labels = self._prepare_training_data(main_file, level_files)
        if features_df is None or labels is None:
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

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

        print(f"\n✅ TRAINING COMPLETE!")
        print(f"   Model: {model_type}")
        print(f"   Test Accuracy: {test_accuracy:.1%}")
        print(f"\n🔍 Top 10 Features:")
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
