"""
🚀 AUTONOMOUS TRADER TRAINER - Level-Based Trading Model Training
================================================================
Trains models for autonomous trading decisions based on multi-timeframe
technical analysis and price interaction with key support/resistance levels.

Training Approach:
1. Historical market simulation with level extraction
2. Label generation based on profitable price movements
3. Classification models for trading actions (buy/sell/hold)
4. Backtesting and validation
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from datetime import datetime
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time

from src.autonomous_trader import (
    AutonomousTrader, MultitimeframeLevelExtractor,
    LevelBasedFeatureEngineer, TradingAction
)
from src.indicator_utils import add_progressive_indicators

warnings.filterwarnings('ignore')


class AutonomousTraderTrainer:
    """
    Trainer for autonomous trading models using multi-timeframe level analysis
    """

    def __init__(self):
        self.level_extractor = MultitimeframeLevelExtractor()
        self.feature_engineer = LevelBasedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_history = []

        # Training parameters
        self.lookforward_periods = [5, 10, 15, 20]  # Periods to look ahead for labeling
        self.profit_threshold = 1.0  # Minimum profit % to consider successful
        self.loss_threshold = -0.5   # Maximum loss % before stop

    def prepare_training_data(self, intraday_data_file: str, level_data_files: Dict[str, str],
                              use_log_scale: bool = True) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Prepare training data by combining intraday prices with multi-timeframe levels

        Args:
            intraday_data_file: Path to intraday data (1h, 4h) for trading decisions
            level_data_files: Dict of higher timeframe data files {'M': path, 'W': path, 'D': path}
            use_log_scale: Whether to use log scale

        Returns:
            Tuple of (features_df, labels_dict)
        """
        print("📊 Preparing training data...")

        # Load intraday data
        with open(intraday_data_file, 'r') as f:
            intraday_json = json.load(f)

        intraday_df = self._json_to_dataframe(intraday_json)
        print(f"✅ Loaded {len(intraday_df)} intraday candles")

        # Extract levels from higher timeframes
        print("🔍 Extracting multi-timeframe levels...")
        levels_by_timeframe = self.level_extractor.extract_levels_from_data(
            level_data_files, use_log_scale
        )

        # Create features for each intraday candle
        print("🛠️ Engineering features...")
        features_list = []
        labels_dict = {action.value: [] for action in TradingAction}

        # Process each candle (skip last few for lookforward labeling)
        max_lookforward = max(self.lookforward_periods)

        for i in range(len(intraday_df) - max_lookforward):
            current_candle = intraday_df.iloc[i]
            current_price = current_candle['close']
            current_volume = current_candle['volume']
            current_time = current_candle.name

            # Create level-based features
            features = self.feature_engineer.create_level_features(
                current_price, current_volume, levels_by_timeframe
            )

            # Add technical indicators
            tech_features = self._calculate_technical_features(intraday_df, i)
            features.update(tech_features)

            # Add timestamp-based features
            time_features = self._calculate_time_features(current_time)
            features.update(time_features)

            features_list.append(features)

            # Generate labels based on future price movements
            future_labels = self._generate_labels(intraday_df, i, current_price)
            for action, label in future_labels.items():
                labels_dict[action].append(label)

            # Progress reporting
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(intraday_df) - max_lookforward} candles")

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_names = list(features_df.columns)

        # Convert labels to numpy arrays
        labels_arrays = {action: np.array(labels) for action, labels in labels_dict.items()}

        print(f"✅ Created {len(features_df)} training samples with {len(self.feature_names)} features")
        return features_df, labels_arrays

    def _json_to_dataframe(self, json_data: Dict) -> pd.DataFrame:
        """Convert JSON candlestick data to DataFrame"""
        candles = json_data.get('candles', [])

        df_data = []
        for candle in candles:
            df_data.append({
                'timestamp': pd.to_datetime(candle['time'], unit='s'),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            })

        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df

    def _calculate_technical_features(self, df: pd.DataFrame, index: int) -> Dict[str, float]:
        """Calculate technical indicator features using existing indicator_utils"""
        # Use a lookback window to get enough data for indicators
        lookback = min(50, index)  # Need more data for TA-Lib indicators
        if lookback < 20:
            return self._get_default_tech_features()

        try:
            # Get subset of data with enough history for indicators
            start_idx = max(0, index - lookback)
            end_idx = index + 1
            df_subset = df.iloc[start_idx:end_idx].copy()

            # Use existing indicator utilities to add all technical indicators
            df_with_indicators = add_progressive_indicators(df_subset)

            # Extract the current (latest) values
            current_row = df_with_indicators.iloc[-1]

            features = {
                'rsi': float(current_row.get('rsi', 50.0)),
                'macd': float(current_row.get('macd', 0.0)),
                'macd_signal': float(current_row.get('macd_signal', 0.0)),
                'bb_position': float(current_row.get('bb_position', 0.5)),
                'volume_ratio': float(current_row.get('volume_ratio', 1.0)),
                'volatility': float(current_row.get('volatility', 0.02)),
                'bb_upper': float(current_row.get('bb_upper', current_row['close'] * 1.02)),
                'bb_lower': float(current_row.get('bb_lower', current_row['close'] * 0.98)),
                'volume_ma20': float(current_row.get('volume_ma20', current_row['volume']))
            }

            # Add some derived features
            current_price = float(current_row['close'])
            features['price'] = current_price
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / current_price * 100
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            return features

        except Exception as e:
            print(f"Warning: Error calculating technical features at index {index}: {e}")
            return self._get_default_tech_features()

    def _get_default_tech_features(self) -> Dict[str, float]:
        """Default technical features when calculation fails"""
        return {
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'bb_position': 0.5,
            'volume_ratio': 1.0,
            'volatility': 0.02,
            'bb_upper': 55000.0,
            'bb_lower': 45000.0,
            'volume_ma20': 1000000.0,
            'price': 50000.0,
            'bb_width': 4.0,
            'macd_histogram': 0.0
        }

    def _calculate_time_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Calculate time-based features"""
        features = {}

        # Hour of day (0-23)
        features['hour'] = timestamp.hour
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)

        # Day of week (0-6)
        features['day_of_week'] = timestamp.dayofweek
        features['is_weekend'] = 1.0 if timestamp.dayofweek >= 5 else 0.0

        # Day of month
        features['day_of_month'] = timestamp.day

        return features

    def _generate_labels(self, df: pd.DataFrame, index: int, current_price: float) -> Dict[str, int]:
        """Generate trading action labels based on future price movements"""
        labels = {}

        # Get future prices for different horizons
        future_prices = {}
        for periods in self.lookforward_periods:
            if index + periods < len(df):
                future_prices[periods] = df['close'].iloc[index + periods]

        if not future_prices:
            # Default to HOLD if no future data
            return {action.value: 0 for action in TradingAction}

        # Analyze future price movements
        best_buy_return = -float('inf')
        best_sell_return = -float('inf')

        for periods, future_price in future_prices.items():
            # Long (buy) return
            buy_return = (future_price - current_price) / current_price * 100
            best_buy_return = max(best_buy_return, buy_return)

            # Short (sell) return
            sell_return = (current_price - future_price) / current_price * 100
            best_sell_return = max(best_sell_return, sell_return)

        # Generate labels based on returns
        if best_buy_return >= self.profit_threshold:
            labels[TradingAction.BUY.value] = 1
        else:
            labels[TradingAction.BUY.value] = 0

        if best_sell_return >= self.profit_threshold:
            labels[TradingAction.SELL.value] = 1
        else:
            labels[TradingAction.SELL.value] = 0

        # HOLD when neither buy nor sell is profitable
        if (best_buy_return < self.profit_threshold and
                best_sell_return < self.profit_threshold):
            labels[TradingAction.HOLD.value] = 1
        else:
            labels[TradingAction.HOLD.value] = 0

        # Close actions (simplified for now)
        labels[TradingAction.CLOSE_LONG.value] = 0
        labels[TradingAction.CLOSE_SHORT.value] = 0

        return labels

    def train_models(self, features_df: pd.DataFrame, labels_dict: Dict[str, np.ndarray],
                     test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict]:
        """
        Train classification models for each trading action

        Args:
            features_df: Features DataFrame
            labels_dict: Dictionary of labels for each action
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Dictionary of training results
        """
        print("🤖 Training autonomous trading models...")

        results = {}

        # Train a model for each action
        for action_name, labels in labels_dict.items():
            if np.sum(labels) < 10:  # Skip actions with too few positive examples
                print(f"⚠️ Skipping {action_name}: insufficient positive examples ({np.sum(labels)})")
                continue

            print(f"\n🎯 Training {action_name} model...")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, test_size=test_size, random_state=random_state,
                stratify=labels if np.sum(labels) > 1 else None
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # GPU detection for XGBoost
            def detect_gpu():
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    return result.returncode == 0
                except:
                    return False

            gpu_available = detect_gpu()

            # Train multiple models and select best
            models_to_try = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=random_state,
                    class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, random_state=random_state
                ),
                'logistic_regression': LogisticRegression(
                    random_state=random_state, class_weight='balanced', max_iter=1000
                )
            }

            # Add XGBoost with GPU if available
            if gpu_available:
                try:
                    models_to_try['xgboost_gpu'] = xgb.XGBClassifier(
                        n_estimators=200, max_depth=8, learning_rate=0.1,
                        random_state=random_state, tree_method='gpu_hist',
                        gpu_id=0, eval_metric='mlogloss'
                    )
                    print(f"   🚀 XGBoost GPU enabled for {action_name}")
                except:
                    pass

            # Add XGBoost CPU as fallback
            models_to_try['xgboost_cpu'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=random_state, tree_method='hist',
                n_jobs=-1, eval_metric='mlogloss'
            )

            best_model = None
            best_score = -1
            best_model_name = None

            for model_name, model in models_to_try.items():
                try:
                    # Time the training for performance comparison
                    start_time = time.time()

                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
                    mean_score = np.mean(cv_scores)

                    training_time = time.time() - start_time

                    gpu_indicator = "🚀" if "gpu" in model_name else "💻" if "xgboost" in model_name else "⚙️"
                    print(f"   {gpu_indicator} {model_name}: CV F1 = {mean_score:.3f} ± {np.std(cv_scores):.3f} ({training_time:.2f}s)")

                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = model
                        best_model_name = model_name

                except Exception as e:
                    print(f"   ❌ {model_name}: Error - {e}")
                    continue

            if best_model is None:
                print(f"❌ No successful model for {action_name}")
                continue

            # Train best model on full training set
            best_model.fit(X_train_scaled, y_train)

            # Evaluate on test set
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)

            # Store results
            self.models[action_name] = best_model
            self.scalers[action_name] = scaler

            results[action_name] = {
                'model_type': best_model_name,
                'cv_score': best_score,
                'test_predictions': y_pred,
                'test_probabilities': y_pred_proba,
                'test_labels': y_test,
                'feature_importance': self._get_feature_importance(best_model, features_df.columns)
            }

            print(f"✅ {action_name}: Best model = {best_model_name} (CV F1 = {best_score:.3f})")

            # Print classification report
            print(f"\n📊 {action_name} Classification Report:")
            print(classification_report(y_test, y_pred))

        self.training_history.append({
            'timestamp': datetime.now(),
            'results': results,
            'feature_count': len(self.feature_names),
            'sample_count': len(features_df)
        })

        return results

    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}

            return dict(zip(feature_names, importances))
        except Exception:
            return {}

    def save_models(self, model_file: str = 'models/autonomous_trader_models.joblib'):
        """Save trained models and scalers"""
        if not self.models:
            print("❌ No models to save")
            return False

        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat(),
                'training_history': self.training_history,
                'profit_threshold': self.profit_threshold,
                'loss_threshold': self.loss_threshold
            }

            joblib.dump(model_data, model_file)
            print(f"✅ Saved autonomous trading models to {model_file}")
            print(f"🎯 Saved {len(self.models)} models: {list(self.models.keys())}")
            return True

        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False

    def plot_training_results(self, results: Dict[str, Dict]):
        """Plot training results and feature importance"""
        if not results:
            print("No results to plot")
            return

        # Create subplots
        n_actions = len(results)
        fig, axes = plt.subplots(2, n_actions, figsize=(5 * n_actions, 10))
        if n_actions == 1:
            axes = axes.reshape(2, 1)

        for i, (action_name, result) in enumerate(results.items()):
            # Confusion matrix
            cm = confusion_matrix(result['test_labels'], result['test_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, i], cmap='Blues')
            axes[0, i].set_title(f'{action_name} Confusion Matrix')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')

            # Feature importance (top 10)
            importance = result['feature_importance']
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                features, importances = zip(*top_features)

                axes[1, i].barh(range(len(features)), importances)
                axes[1, i].set_yticks(range(len(features)))
                axes[1, i].set_yticklabels(features)
                axes[1, i].set_title(f'{action_name} Top Features')
                axes[1, i].invert_yaxis()

        plt.tight_layout()
        plt.savefig('autonomous_trader_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def backtest_models(self, test_data_file: str, level_data_files: Dict[str, str],
                        initial_balance: float = 10000.0) -> Dict[str, float]:
        """
        Simple backtest of the trained models

        Args:
            test_data_file: Intraday data for backtesting
            level_data_files: Level data files
            initial_balance: Starting balance

        Returns:
            Backtest results
        """
        print("📈 Running backtest...")

        # Create autonomous trader instance
        trader = AutonomousTrader()
        trader.models = self.models
        trader.scalers = self.scalers
        trader.feature_names = self.feature_names

        # Update levels
        trader.update_levels(level_data_files, force_update=True)

        # Load test data
        with open(test_data_file, 'r') as f:
            test_json = json.load(f)
        test_df = self._json_to_dataframe(test_json)

        # Simple backtest simulation
        balance = initial_balance
        position = None  # None, 'long', 'short'
        entry_price = None
        trades = []

        for i in range(100, len(test_df)):  # Start after some data
            current_price = test_df['close'].iloc[i]
            current_volume = test_df['volume'].iloc[i]

            # Get trading signal
            signal = trader.make_trading_decision(current_price, current_volume)

            # Execute trades (simplified)
            if signal.action == TradingAction.BUY and position is None:
                position = 'long'
                entry_price = current_price
                print(f"🟢 BUY at ${current_price:.2f} (confidence: {signal.confidence:.1%})")

            elif signal.action == TradingAction.SELL and position is None:
                position = 'short'
                entry_price = current_price
                print(f"🔴 SELL at ${current_price:.2f} (confidence: {signal.confidence:.1%})")

            elif position == 'long' and (signal.action == TradingAction.SELL or
                                         signal.action == TradingAction.CLOSE_LONG):
                # Close long position
                pnl = (current_price - entry_price) / entry_price * balance
                balance += pnl
                trades.append(pnl)
                print(f"📤 CLOSE LONG at ${current_price:.2f}, P&L: ${pnl:.2f}")
                position = None
                entry_price = None

            elif position == 'short' and (signal.action == TradingAction.BUY or
                                          signal.action == TradingAction.CLOSE_SHORT):
                # Close short position
                pnl = (entry_price - current_price) / entry_price * balance
                balance += pnl
                trades.append(pnl)
                print(f"📤 CLOSE SHORT at ${current_price:.2f}, P&L: ${pnl:.2f}")
                position = None
                entry_price = None

        # Backtest results
        total_return = (balance - initial_balance) / initial_balance * 100
        win_rate = sum(1 for trade in trades if trade > 0) / len(trades) * 100 if trades else 0
        avg_trade = np.mean(trades) if trades else 0

        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return_pct': total_return,
            'number_of_trades': len(trades),
            'win_rate_pct': win_rate,
            'average_trade': avg_trade
        }

        print("\n📊 Backtest Results:")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Final Balance: ${balance:,.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Number of Trades: {len(trades)}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Average Trade: ${avg_trade:.2f}")

        return results


def train_autonomous_trading_system():
    """Main training function"""
    print("🚀 Training Autonomous Trading System")
    print("=" * 50)

    # Initialize trainer
    trainer = AutonomousTraderTrainer()

    # Data files
    intraday_data = 'data/BTCUSDT-1h.json'  # 1-hour data for trading decisions
    level_data_files = {
        'M': 'data/BTCUSDT-M.json',
        'W': 'data/BTCUSDT-W.json',
        'D': 'data/BTCUSDT-D.json'
    }

    try:
        # Prepare training data
        features_df, labels_dict = trainer.prepare_training_data(
            intraday_data, level_data_files
        )

        # Train models
        results = trainer.train_models(features_df, labels_dict)

        # Save models
        trainer.save_models()

        # Plot results
        trainer.plot_training_results(results)

        # Run backtest
        trainer.backtest_models(intraday_data, level_data_files)

        print("\n✅ Training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    train_autonomous_trading_system()
