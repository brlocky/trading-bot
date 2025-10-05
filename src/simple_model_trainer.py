"""
🚀 SIMPLE MODEL TRAINER - Clean and Reusable Training System
===========================================================
Handles all the complexity of model training, data preparation, and predictions
for the Simple_Model_Debug.ipynb notebook.

This class encapsulates:
- Data preparation and cleaning
- Label processing and encoding
- GPU model training (with CPU fallback)
- Model saving/loading
- Feature engineering for predictions
- Prediction generation with confidence scores
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import time
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

from src.autonomous_trader_trainer import AutonomousTraderTrainer
from src.autonomous_trader import AutonomousTrader

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
        print("📊 Preparing training data...")

        try:
            # Get raw features and labels
            features_df, labels_dict = self.trainer.prepare_training_data(main_file, level_files)

            if features_df is None or len(features_df) == 0:
                print("❌ No training data prepared")
                return None, None

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

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _train_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train_encoded: np.ndarray, y_test_encoded: np.ndarray) -> Tuple[Any, str, Dict]:
        """Train model with GPU only - no CPU fallback"""
        gpu_available = self._detect_gpu()
        print(f"🖥️  GPU Detection: {'✅ Available' if gpu_available else '❌ Not Available'}")
        
        if not gpu_available:
            raise RuntimeError("❌ GPU not available! Training requires GPU. Please ensure NVIDIA GPU and drivers are properly installed.")
        
        results = {'training_time': None, 'device': 'GPU'}
        
        print("   � Training with GPU...")
        start_time = time.time()
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, tree_method='gpu_hist', gpu_id=0,
            eval_metric='mlogloss'
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


class SimpleModelPredictor:
    """
    Simple prediction system for testing trained models
    """

    def __init__(self, model_trainer: SimpleModelTrainer):
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
                             data_folder: str = 'data_test') -> Optional[pd.DataFrame]:
        """Generate predictions for a symbol"""
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

                # Create features
                features = self.trader.feature_engineer.create_level_features(
                    float(row['close']), float(row['volume']), self.trader.current_levels
                )

                # Convert to DataFrame and ensure all required features
                feature_df = pd.DataFrame([features])
                for col in feature_columns:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0

                feature_df = feature_df[feature_columns]

                # Make prediction
                prediction_encoded = trained_model.predict(feature_df)[0]
                confidence = trained_model.predict_proba(feature_df).max()
                prediction = label_encoder.inverse_transform([prediction_encoded])[0]

                signals.append({
                    'datetime': row['datetime'],
                    'close': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'open': row['open'] if pd.notna(row['open']) else row['close'],
                    'volume': row['volume'],
                    'action': prediction,
                    'confidence': confidence,
                    'reasoning': f'{model_type} ({main_timeframe}): {prediction} ({confidence:.1%})'
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
