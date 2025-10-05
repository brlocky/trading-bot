"""
Autonomous Trader - Main trading decision system combining multi-timeframe analysis
"""

import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.core.trading_types import TradingAction, TradingSignal
from src.extraction.level_extractor import MultitimeframeLevelExtractor
from src.extraction.feature_engineer import LevelBasedFeatureEngineer


class AutonomousTrader:
    """
    Main autonomous trading system that combines multi-timeframe analysis
    with level-based feature engineering to make trading decisions
    """

    def __init__(self):
        self.level_extractor = MultitimeframeLevelExtractor()
        self.feature_engineer = LevelBasedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.current_levels = {}
        self.last_update = None

    def load_models(self, model_file: str = 'models/autonomous_trader_models.joblib') -> bool:
        """Load trained autonomous trading models"""
        try:
            model_data = joblib.load(model_file)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']

            print(f"✅ Loaded autonomous trading models")
            print(f"🎯 Available actions: {list(self.models.keys())}")
            print(f"🔧 Features: {len(self.feature_names)}")
            return True

        except FileNotFoundError:
            print(f"❌ Model file not found: {model_file}")
            print("💡 Train models first using the autonomous trader trainer")
            return False
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def update_levels(self, data_files: Dict[str, str], force_update: bool = False) -> bool:
        """Update level information from multiple timeframes"""
        try:
            # Check if update is needed (avoid frequent re-computation)
            current_time = datetime.now()
            if (not force_update and self.last_update and
                    (current_time - self.last_update).total_seconds() < 3600):  # 1 hour cache
                return True

            print("🔄 Updating multi-timeframe levels...")
            self.current_levels = self.level_extractor.extract_levels_from_data(data_files)
            self.last_update = current_time

            total_levels = sum(len(levels) for levels in self.current_levels.values())
            print(f"✅ Updated {total_levels} levels across {len(self.current_levels)} timeframes")
            return True

        except Exception as e:
            print(f"❌ Error updating levels: {e}")
            return False

    def make_trading_decision(self, current_price: float, current_volume: float = 1000000,
                              additional_features: Optional[Dict[str, float]] = None) -> TradingSignal:
        """
        Make autonomous trading decision based on current market conditions and levels

        Args:
            current_price: Current market price
            current_volume: Current trading volume
            additional_features: Additional technical indicators (RSI, etc.)

        Returns:
            TradingSignal with action and reasoning
        """
        if not self.models:
            return TradingSignal(
                action=TradingAction.HOLD,
                confidence=0.0,
                reasoning="No models loaded"
            )

        if not self.current_levels:
            return TradingSignal(
                action=TradingAction.HOLD,
                confidence=0.0,
                reasoning="No level data available"
            )

        try:
            # Create level-based features
            features = self.feature_engineer.create_level_features(
                current_price, current_volume, self.current_levels
            )

            # Add additional features if provided
            if additional_features:
                features.update(additional_features)

            # Convert to DataFrame for model input
            feature_df = pd.DataFrame([features])

            # Ensure all required features are present
            for feature_name in self.feature_names:
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = 0.0

            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]

            # Make predictions for each action
            action_probabilities = {}
            for action_name, model in self.models.items():
                if action_name in self.scalers:
                    X_scaled = self.scalers[action_name].transform(feature_df)
                    prob = model.predict_proba(X_scaled)[0]

                    # Get probability for positive class (action should be taken)
                    action_probabilities[action_name] = prob[1] if len(prob) > 1 else prob[0]

            # Determine best action
            best_action = max(action_probabilities.items(), key=lambda x: x[1])
            action_name, confidence = best_action

            # Convert action name to TradingAction enum
            action = TradingAction(action_name)

            # Calculate entry/exit levels based on nearby support/resistance
            entry_price, stop_loss, take_profit = self._calculate_trade_levels(
                current_price, action, features
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(action, confidence, features)

            return TradingSignal(
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                risk_reward_ratio=self._calculate_risk_reward(entry_price, stop_loss, take_profit)
            )

        except Exception as e:
            print(f"❌ Error making trading decision: {e}")
            return TradingSignal(
                action=TradingAction.HOLD,
                confidence=0.0,
                reasoning=f"Error in decision making: {e}"
            )

    def _calculate_trade_levels(self, current_price: float, action: TradingAction,
                                features: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate entry, stop loss, and take profit levels"""
        if action == TradingAction.HOLD:
            return None, None, None

        entry_price = current_price

        # Use support/resistance distances for stop loss and take profit
        support_distance = features.get('distance_to_support', 2.0)
        resistance_distance = features.get('distance_to_resistance', 2.0)

        if action == TradingAction.BUY:
            # Stop loss below support, take profit near resistance
            stop_loss = current_price * (1 - (support_distance + 0.5) / 100)
            take_profit = current_price * (1 + (resistance_distance - 0.2) / 100)

        elif action == TradingAction.SELL:
            # Stop loss above resistance, take profit near support
            stop_loss = current_price * (1 + (resistance_distance + 0.5) / 100)
            take_profit = current_price * (1 - (support_distance - 0.2) / 100)

        else:
            stop_loss = None
            take_profit = None

        return entry_price, stop_loss, take_profit

    def _calculate_risk_reward(self, entry_price: Optional[float],
                               stop_loss: Optional[float],
                               take_profit: Optional[float]) -> Optional[float]:
        """Calculate risk-reward ratio"""
        if not all([entry_price, stop_loss, take_profit]):
            return None

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        return reward / risk if risk > 0 else None

    def _generate_reasoning(self, action: TradingAction, confidence: float,
                            features: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the trading decision"""
        reasoning_parts = []

        # Action and confidence
        reasoning_parts.append(f"Action: {action.value.upper()} (confidence: {confidence:.1%})")

        # Level analysis
        support_dist = features.get('distance_to_support', 0)
        resistance_dist = features.get('distance_to_resistance', 0)

        if support_dist < 1.0:
            reasoning_parts.append(f"Near support level ({support_dist:.2f}% away)")
        if resistance_dist < 1.0:
            reasoning_parts.append(f"Near resistance level ({resistance_dist:.2f}% away)")

        # Strength analysis
        support_strength = features.get('support_strength', 0)
        resistance_strength = features.get('resistance_strength', 0)

        if support_strength > 0.8:
            reasoning_parts.append("Strong support detected")
        if resistance_strength > 0.8:
            reasoning_parts.append("Strong resistance detected")

        # Volume context
        volume_norm = features.get('volume_normalized', 1.0)
        if volume_norm > 2.0:
            reasoning_parts.append("High volume environment")
        elif volume_norm < 0.5:
            reasoning_parts.append("Low volume environment")

        return " | ".join(reasoning_parts)

    def get_current_market_context(self) -> Dict[str, any]:
        """Get current market context including levels and features"""
        if not self.current_levels:
            return {"error": "No level data available"}

        context = {
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "timeframes": list(self.current_levels.keys()),
            "total_levels": sum(len(levels) for levels in self.current_levels.values()),
            "levels_by_timeframe": {}
        }

        for timeframe, levels in self.current_levels.items():
            context["levels_by_timeframe"][timeframe] = {
                "count": len(levels),
                "types": list(set(level.level_type for level in levels)),
                "price_range": {
                    "min": min(level.price for level in levels) if levels else None,
                    "max": max(level.price for level in levels) if levels else None
                }
            }

        return context


def test_autonomous_trader():
    """Test the autonomous trading system"""
    print("🧪 Testing Autonomous Trading System")
    print("=" * 50)

    # Initialize trader
    trader = AutonomousTrader()

    # Test level extraction with sample data files
    data_files = {
        'M': 'data/BTCUSDT-M.json',
        'W': 'data/BTCUSDT-W.json',
        'D': 'data/BTCUSDT-D.json'
    }

    # Update levels
    success = trader.update_levels(data_files, force_update=True)
    if not success:
        print("❌ Failed to update levels")
        return False

    # Get market context
    context = trader.get_current_market_context()
    print(f"📊 Market Context: {context}")

    # Test trading decision (without trained models)
    current_price = 65000.0  # Sample BTC price
    signal = trader.make_trading_decision(current_price, 2000000)

    print(f"\n🎯 Trading Signal:")
    print(f"   Action: {signal.action.value}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Entry: ${signal.entry_price:,.2f}" if signal.entry_price else "   Entry: N/A")
    print(f"   Stop Loss: ${signal.stop_loss:,.2f}" if signal.stop_loss else "   Stop Loss: N/A")
    print(f"   Take Profit: ${signal.take_profit:,.2f}" if signal.take_profit else "   Take Profit: N/A")
    print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}" if signal.risk_reward_ratio else "   Risk/Reward: N/A")
    print(f"   Reasoning: {signal.reasoning}")

    return True


if __name__ == "__main__":
    test_autonomous_trader()
