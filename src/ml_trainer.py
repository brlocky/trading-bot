"""
🤖 ML TRAINER - Simple Machine Learning Training System (KISS Principle)
======================================================================
Simple XGBoost models for entry, stop-loss, and take_profit prediction
Features: 7 essential features, direct CSV training, clean and simple
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from datetime import datetime
import warnings
import os
import json

warnings.filterwarnings('ignore')


class MLTrainer:
    def save_models(self, filename='models/progressive_ml_models.joblib'):
        """Save progressive trained models"""
        if not self.models:
            print("❌ No models to save. Train models first.")
            return None

        # Create models directory if it doesn't exist
        model_dir = os.path.dirname(filename)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'test_results': self.test_results,
            'timestamp': datetime.now(),
            'model_type': 'progressive_models'
        }

        try:
            joblib.dump(model_data, filename)
            print(f"💾 Progressive models saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return None
    """Simple ML training system for trading predictions - KISS principle"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_data = None
        self.test_results = {}
        self.trade_memory = []  # Progressive trade memory
        self.max_memory = 20

    def generate_progressive_trades(self, df, max_trades=30):
        """Generate trades from progressive market data"""
        print("🔍 Generating Progressive Trades...")

        trades = []

        # Simple progressive signal detection
        for i in range(50, len(df)):
            current = df.iloc[i]

            # Simple signals with volume confirmation
            rsi_oversold = current['rsi'] < 30 and current['volume_ratio'] > 1.2
            rsi_overbought = current['rsi'] > 70 and current['volume_ratio'] > 1.2

            if rsi_oversold:
                signal_type = 'LONG'
                entry_price = current['close']
                stop_loss = entry_price * 0.975
                take_profit = entry_price * 1.05

            elif rsi_overbought:
                signal_type = 'SHORT'
                entry_price = current['close']
                stop_loss = entry_price * 1.025
                take_profit = entry_price * 0.95
            else:
                continue

            # Simulate trade exit
            exit_result = self._simulate_progressive_exit(df, i, entry_price, stop_loss, take_profit, signal_type)

            if exit_result:
                trade = {
                    'timestamp': df.index[i],
                    'strategy': f'PROGRESSIVE_{signal_type}',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_price': exit_result['exit_price'],
                    'pnl_pct': exit_result['pnl_pct'],
                    'volume': current['volume'],
                    'volatility': current['volatility'],
                    'rsi': current['rsi'],
                    'bb_position': current['bb_position'],
                    'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                }

                trades.append(trade)

                # Add to trade memory
                self.trade_memory.append({
                    'pnl_pct': exit_result['pnl_pct'],
                    'timestamp': df.index[i]
                })
                if len(self.trade_memory) > self.max_memory:
                    self.trade_memory = self.trade_memory[-self.max_memory:]

                if len(trades) >= max_trades:
                    break

        print(f"✅ Generated {len(trades)} progressive trades")
        return trades

    def _simulate_progressive_exit(self, df, entry_idx, entry_price, stop_loss, take_profit, signal_type):
        """Simulate trade exit for progressive trades"""
        max_check = min(entry_idx + 24, len(df))  # Check next 24 hours

        for i in range(entry_idx + 1, max_check):
            candle = df.iloc[i]
            high, low = candle['high'], candle['low']

            if signal_type == 'LONG':
                if low <= stop_loss:
                    return {'exit_price': stop_loss, 'pnl_pct': (stop_loss - entry_price) / entry_price * 100}
                elif high >= take_profit:
                    return {'exit_price': take_profit, 'pnl_pct': (take_profit - entry_price) / entry_price * 100}
            else:  # SHORT
                if high >= stop_loss:
                    return {'exit_price': stop_loss, 'pnl_pct': (entry_price - stop_loss) / entry_price * 100}
                elif low <= take_profit:
                    return {'exit_price': take_profit, 'pnl_pct': (entry_price - take_profit) / entry_price * 100}

        # Timeout exit
        final_price = df.iloc[max_check - 1]['close']
        if signal_type == 'LONG':
            pnl_pct = (final_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - final_price) / entry_price * 100
        return {'exit_price': final_price, 'pnl_pct': pnl_pct}

    def create_progressive_features(self):
        """Create features with trade memory - enhanced version"""
        print("🔧 Creating Progressive Features with Trade Memory...")

        if self.training_data is None:
            raise ValueError("Load training data first")

        df = self.training_data.copy()

        # 10 Enhanced Features (7 original + 3 memory)
        features = pd.DataFrame({
            'entry_price': df['entry_price'],
            'volume': df['volume'],
            'volatility': df['volatility'],
            'rsi': df['rsi'],
            'bb_position': df['bb_position'],
            'risk_pct': abs(df['entry_price'] - df['stop_loss']) / df['entry_price'],
            'reward_pct': abs(df['take_profit'] - df['entry_price']) / df['entry_price'],

            # Trade memory features (simulated for training)
            'memory_win_rate': np.random.uniform(0.3, 0.7, len(df)),
            'memory_avg_pnl': np.random.uniform(-1, 2, len(df)),
            'consecutive_wins': np.random.randint(0, 5, len(df))
        })

        features = features.fillna(0)
        self.feature_names = features.columns.tolist()

        print(f"✅ Created {len(self.feature_names)} progressive features")
        return features

    def train_progressive_mode(self):
        """Train models using progressive market data and trade memory"""
        print("🚀 Training Progressive Mode...")

        # Load market data and generate trades
        df = self.load_market_data_progressive()
        if df.empty:
            print("❌ Failed to load market data")
            return False

        trades = self.generate_progressive_trades(df)
        if not trades:
            print("❌ No trades generated")
            return False

        # Convert to training data
        self.training_data = pd.DataFrame(trades)

        # Train with progressive features
        X = self.create_progressive_features()
        y = pd.DataFrame({
            'entry_price': self.training_data['entry_price'],
            'stop_loss': self.training_data['stop_loss'],
            'take_profit': self.training_data['take_profit']
        })

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for target in ['entry_price', 'stop_loss', 'take_profit']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train_scaled, y_train[target])

            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test[target], y_pred)

            self.models[target] = model
            self.scalers[target] = scaler
            self.test_results[target] = {'r2': r2}

            print(f"   ✅ Progressive {target}: R² = {r2:.3f}")

        return True

    def load_market_data_progressive(self, file_path='data/BTCUSDT-1h.json'):
        """Load market data and add progressive TA-Lib indicators"""
        print("📊 Loading Market Data for Progressive Training...")
        try:
            from indicator_utils import add_progressive_indicators
            with open(file_path, 'r') as f:
                data = json.load(f)
            candles = data.get('candles', data)
            df = pd.DataFrame(candles)
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            df = add_progressive_indicators(df)
            print(f"✅ Loaded {len(df)} candles with progressive indicators")
            return df
        except Exception as e:
            print(f"❌ Error loading market data: {e}")
            return pd.DataFrame()


def main():
    """Enhanced main function - KISS principle with progressive option"""
    trainer = MLTrainer()

    print("🤖 ML TRAINER - Progressive Mode Only")
    print("=" * 60)
    print("🔧 Running Progressive Mode (TA-Lib + Trade Memory)...")
    if trainer.train_progressive_mode():
        trainer.save_models('models/progressive_ml_models.joblib')
        print("\n🎯 PROGRESSIVE ML RESULTS")
        print("=" * 40)
        for target, results in trainer.test_results.items():
            r2 = results['r2']
            status = "🔥 EXCELLENT" if r2 > 0.8 else "✅ GOOD" if r2 > 0.6 else "⚠️ FAIR"
            print(f"   {target}: R² = {r2:.3f} {status}")
        avg_r2 = np.mean([r['r2'] for r in trainer.test_results.values()])
        print(f"\n📊 Average R²: {avg_r2:.3f}")
        print(f"🧠 Trade Memory: {len(trainer.trade_memory)} trades")
        print("🎉 SUCCESS! Progressive models ready!")
    else:
        print("❌ Progressive training failed. No models generated.")
    return trainer


if __name__ == "__main__":
    main()
