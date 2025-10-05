"""
🔮 ML PREDICTOR - Simple Prediction System (KISS Principle)
=========================================================
Simple prediction system using 7 essential features
Clean, simple, and effective
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import talib
import json

warnings.filterwarnings('ignore')


class MLPredictor:
    """Simple ML prediction system - KISS principle"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.trade_memory = []  # Progressive trade memory
        self.is_progressive = False

    def load_models(self, model_file='models/simple_ml_models.joblib'):
        """Load simple trained models"""
        print("🤖 Loading Simple ML Models...")

        try:
            model_data = joblib.load(model_file)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']

            print(f"✅ Loaded simple models: {list(self.models.keys())}")
            print(f"🔧 Features: {len(self.feature_names)}")
            print(f"📅 Training date: {model_data.get('timestamp', 'Unknown')}")
            return True

        except FileNotFoundError:
            print(f"❌ Model file not found: {model_file}")
            print("💡 Train models first using ml_trainer.py")
            return False
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def create_prediction_features(self, entry_price, volume, volatility, rsi, bb_position, 
                                   stop_loss, take_profit):
        """Create the same 7 features used in training"""
        
        # Calculate risk and reward percentages
        risk_pct = abs(entry_price - stop_loss) / entry_price
        reward_pct = abs(take_profit - entry_price) / entry_price
        
        # Create feature vector in same order as training
        features = {
            'entry_price': entry_price,
            'volume': volume,
            'volatility': volatility,
            'rsi': rsi,
            'bb_position': bb_position,
            'risk_pct': risk_pct,
            'reward_pct': reward_pct
        }
        
        # Convert to DataFrame for consistency
        feature_df = pd.DataFrame([features])
        
        return feature_df

    def make_simple_prediction(self, entry_price, volume=1000000, volatility=0.02, 
                               rsi=50, bb_position=0.5, stop_loss=None, take_profit=None):
        """Make simple prediction with basic inputs"""
        
        if not self.models:
            print("❌ No models loaded. Load models first.")
            return None
            
        # Use defaults if not provided
        if stop_loss is None:
            stop_loss = entry_price * 0.975  # 2.5% stop loss
        if take_profit is None:
            take_profit = entry_price * 1.05   # 5% take profit
            
        print(f"🔮 Making Simple Prediction...")
        print(f"   📊 Entry: ${entry_price:,.2f}")
        print(f"   📈 Volume: {volume:,}")
        print(f"   📊 RSI: {rsi:.1f}")
        print(f"   🎯 BB Position: {bb_position:.2f}")
        
        try:
            # Create features
            features = self.create_prediction_features(
                entry_price, volume, volatility, rsi, bb_position, stop_loss, take_profit
            )
            
            predictions = {}
            
            # Make predictions for each target
            for target in ['entry_price', 'stop_loss', 'take_profit']:
                if target in self.models and target in self.scalers:
                    # Scale features
                    X_scaled = self.scalers[target].transform(features)
                    
                    # Predict
                    pred = self.models[target].predict(X_scaled)[0]
                    predictions[target] = pred
            
            print(f"✅ Simple prediction SUCCESS!")
            print(f"   📊 Entry: ${predictions.get('entry_price', entry_price):,.2f}")
            print(f"   🛡️ Stop: ${predictions.get('stop_loss', stop_loss):,.2f}")
            print(f"   🎯 Profit: ${predictions.get('take_profit', take_profit):,.2f}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def predict_from_current_data(self, market_data):
        """Make prediction from current market data"""
        
        if not isinstance(market_data, dict):
            print("❌ Market data must be a dictionary")
            return None
            
        # Extract required fields with defaults
        entry_price = market_data.get('close', market_data.get('entry_price', 100000))
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        rsi = market_data.get('rsi', 50)
        bb_position = market_data.get('bb_position', 0.5)
        
        return self.make_simple_prediction(entry_price, volume, volatility, rsi, bb_position)

    def test_prediction_system(self):
        """Test the prediction system with sample data"""
        print("🧪 Testing Simple Prediction System")
        print("=" * 50)
        
        if not self.models:
            print("❌ No models loaded")
            return False
            
        # Test with sample BTC data
        test_cases = [
            {"entry_price": 117000, "rsi": 30, "bb_position": 0.2, "volume": 2000000, "name": "Oversold Signal"},
            {"entry_price": 115000, "rsi": 70, "bb_position": 0.8, "volume": 1500000, "name": "Overbought Signal"},
            {"entry_price": 116000, "rsi": 50, "bb_position": 0.5, "volume": 1000000, "name": "Neutral Signal"}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 Test Case {i}: {test_case['name']}")
            prediction = self.make_simple_prediction(
                entry_price=test_case['entry_price'],
                volume=test_case['volume'],
                rsi=test_case['rsi'],
                bb_position=test_case['bb_position']
            )
            
            if prediction:
                # Calculate risk/reward
                entry = prediction.get('entry_price', test_case['entry_price'])
                stop = prediction.get('stop_loss', entry * 0.975)
                profit = prediction.get('take_profit', entry * 1.05)
                
                risk = abs(entry - stop) / entry * 100
                reward = abs(profit - entry) / entry * 100
                rr_ratio = reward / risk if risk > 0 else 0
                
                print(f"   💡 Risk: {risk:.1f}% | Reward: {reward:.1f}% | RR: {rr_ratio:.1f}")
        
        print("\n✅ Prediction system test complete!")
        return True

    def load_progressive_models(self, model_file='models/progressive_ml_models.joblib'):
        """Load progressive models with trade memory"""
        print("🤖 Loading Progressive ML Models...")

        try:
            model_data = joblib.load(model_file)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            
            # Load trade memory if available
            self.trade_memory = model_data.get('trade_memory', [])
            self.is_progressive = True

            print(f"✅ Loaded progressive models: {list(self.models.keys())}")
            print(f"🔧 Features: {len(self.feature_names)}")
            print(f"🧠 Trade memory: {len(self.trade_memory)} trades")
            return True

        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"❌ Error loading progressive models: {e}")
            return False

    def get_trade_memory_features(self):
        """Get trade memory features for progressive predictions"""
        if not self.trade_memory:
            return {
                'memory_win_rate': 0.5,
                'memory_avg_pnl': 0.0,
                'consecutive_wins': 0
            }
            
        recent_trades = self.trade_memory[-10:]  # Last 10 trades
        
        wins = [t for t in recent_trades if t['pnl_pct'] > 0]
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0.5
        avg_pnl = sum(t['pnl_pct'] for t in recent_trades) / len(recent_trades) if recent_trades else 0.0
        
        # Count consecutive wins from end
        consecutive_wins = 0
        for trade in reversed(recent_trades):
            if trade['pnl_pct'] > 0:
                consecutive_wins += 1
            else:
                break
                
        return {
            'memory_win_rate': win_rate,
            'memory_avg_pnl': avg_pnl,
            'consecutive_wins': consecutive_wins
        }

    def make_progressive_prediction(self, market_data):
        """Make prediction with progressive features and trade memory"""
        
        if not self.models or not self.is_progressive:
            print("❌ No progressive models loaded")
            return None
            
        print("🔮 Making Progressive Prediction...")
        print(f"   📊 Entry: ${market_data['close']:,.2f}")
        print(f"   🧠 Trade Memory: {len(self.trade_memory)} trades")
        
        # Show memory stats
        if self.trade_memory:
            memory_stats = self.get_trade_memory_features()
            print(f"   💡 Memory Win Rate: {memory_stats['memory_win_rate']:.1%}")
            print(f"   💡 Avg P&L: {memory_stats['memory_avg_pnl']:+.2f}%")
        
        try:
            # Create progressive features
            features = {
                'entry_price': market_data['close'],
                'volume': market_data.get('volume', 1000000),
                'volatility': market_data.get('volatility', 0.02),
                'rsi': market_data.get('rsi', 50),
                'bb_position': market_data.get('bb_position', 0.5),
                'risk_pct': 0.025,  # Default 2.5%
                'reward_pct': 0.05,  # Default 5%
            }
            
            # Add trade memory features
            memory_features = self.get_trade_memory_features()
            features.update(memory_features)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            predictions = {}
            
            # Make predictions
            for target in ['entry_price', 'stop_loss', 'take_profit']:
                if target in self.models and target in self.scalers:
                    X_scaled = self.scalers[target].transform(feature_df)
                    pred = self.models[target].predict(X_scaled)[0]
                    predictions[target] = pred
            
            print("✅ Progressive prediction SUCCESS!")
            print(f"   📊 Entry: ${predictions.get('entry_price', market_data['close']):,.2f}")
            print(f"   🛡️ Stop: ${predictions.get('stop_loss', market_data['close'] * 0.975):,.2f}")
            print(f"   🎯 Profit: ${predictions.get('take_profit', market_data['close'] * 1.05):,.2f}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Progressive prediction error: {e}")
            return None

    def update_trade_memory(self, trade_result):
        """Update trade memory with new trade result"""
        self.trade_memory.append({
            'pnl_pct': trade_result['pnl_pct'],
            'timestamp': trade_result.get('timestamp', datetime.now())
        })
        
        # Keep only last 20 trades
        if len(self.trade_memory) > 20:
            self.trade_memory = self.trade_memory[-20:]
        
        print(f"🧠 Trade memory updated: {trade_result['pnl_pct']:+.2f}% (Total: {len(self.trade_memory)})")


def main():
    """Enhanced main function - KISS principle with progressive option"""
    predictor = MLPredictor()

    print("🔮 ML PREDICTOR - KISS Principle (Simple + Progressive)")
    print("=" * 60)
    
    # Try progressive models first
    if predictor.load_progressive_models('models/progressive_ml_models.joblib'):
        print("🚀 Progressive Mode Active!")
        
        # Test with current market conditions
        current_btc_data = {
            'close': 117391.22,
            'volume': 1800000,
            'volatility': 0.025,
            'rsi': 45.5,
            'bb_position': 0.65
        }
        
        prediction = predictor.make_progressive_prediction(current_btc_data)
        
        if prediction:
            print("🎉 Progressive prediction ready for live trading!")
        else:
            print("❌ Progressive prediction failed")
            
    # Fallback to simple models
    elif predictor.load_models('models/simple_ml_models.joblib'):
        print("🔄 Using Simple Mode (fallback)")
        
        predictor.test_prediction_system()
        
        current_btc_data = {
            'close': 117391.22,
            'volume': 1800000,
            'volatility': 0.025,
            'rsi': 45.5,
            'bb_position': 0.65
        }
        
        prediction = predictor.predict_from_current_data(current_btc_data)
        
        if prediction:
            print("🎉 Simple prediction ready!")
        else:
            print("❌ Simple prediction failed")
    else:
        print("❌ No models found")
        print("💡 Run ml_trainer.py first to train models")
    
    return predictor


if __name__ == "__main__":
    main()