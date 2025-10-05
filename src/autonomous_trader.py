"""
🤖 AUTONOMOUS TRADER - Level-Based Trading System
=====================================================
An autonomous trading system that learns to make trading decisions based on:
- Multi-timeframe technical analysis (M, W, D levels)
- Price interaction with support/resistance levels
- Volume profile key levels (POC, VAH, VAL)
- Channel boundaries and pivot points

This system moves beyond price prediction to autonomous action learning.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

from src.ta.technical_analysis import TechnicalAnalysisProcessor, AnalysisDict, ChartInterval
from src.ta.middlewares.zigzag import zigzag_middleware
from src.ta.middlewares.volume_profile import volume_profile_middleware
from src.ta.middlewares.channels import channels_middleware
from src.ta.middlewares.levels import levels_middleware

warnings.filterwarnings('ignore')


class TradingAction(Enum):
    """Trading actions the model can take"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradingSignal:
    """Trading signal with reasoning"""
    action: TradingAction
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    risk_reward_ratio: Optional[float] = None


@dataclass
class LevelInfo:
    """Information about a support/resistance level"""
    price: float
    strength: float  # How many times it was tested
    distance: float  # Distance from current price (%)
    level_type: str  # 'support', 'resistance', 'poc', 'vah', 'val', etc.
    timeframe: ChartInterval
    last_test_time: Optional[pd.Timestamp] = None


class MultitimeframeLevelExtractor:
    """Extracts key levels from multiple timeframes"""

    def __init__(self):
        self.higher_timeframes: List[ChartInterval] = ['M', 'W', 'D']  # Monthly, Weekly, Daily
        self.level_cache = {}  # Cache for extracted levels

    def extract_levels_from_data(self, data_files: Dict[str, str],
                                 use_log_scale: bool = True) -> Dict[ChartInterval, List[LevelInfo]]:
        """
        Extract key levels from multiple timeframe data files

        Args:
            data_files: Dict mapping timeframe to file path {'M': 'path/to/monthly.json', ...}
            use_log_scale: Whether to use logarithmic scale

        Returns:
            Dict mapping timeframe to list of LevelInfo objects
        """
        all_levels = {}

        for timeframe, file_path in data_files.items():
            if timeframe not in self.higher_timeframes:
                continue

            try:
                # Load and process data
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                # Convert to DataFrame
                df = self._json_to_dataframe(json_data)

                # Run technical analysis
                processor = TechnicalAnalysisProcessor(df, timeframe, use_log_scale)
                processor.register_middleware(zigzag_middleware)
                processor.register_middleware(volume_profile_middleware)
                processor.register_middleware(channels_middleware)
                processor.register_middleware(levels_middleware)

                analysis = processor.run()

                # Extract levels from analysis
                levels = self._extract_levels_from_analysis(analysis, timeframe, df)
                all_levels[timeframe] = levels

                print(f"✅ Extracted {len(levels)} levels from {timeframe} timeframe")

            except Exception as e:
                print(f"❌ Error processing {timeframe} data: {e}")
                all_levels[timeframe] = []

        return all_levels

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

    def _extract_levels_from_analysis(self, analysis: AnalysisDict,
                                      timeframe: ChartInterval,
                                      df: pd.DataFrame) -> List[LevelInfo]:
        """Extract LevelInfo objects from technical analysis results"""
        levels = []
        current_price = df['close'].iloc[-1]

        # Extract levels from different middlewares
        for middleware_name, results in analysis.items():

            # Volume Profile levels (POC, VAH, VAL)
            if 'lines' in results and middleware_name == 'volume_profile_periods':
                levels.extend(self._extract_volume_profile_levels(
                    results['lines'], timeframe, current_price
                ))

            # Support/Resistance levels from levels middleware
            if 'lines' in results and middleware_name == 'levels':
                levels.extend(self._extract_support_resistance_levels(
                    results['lines'], timeframe, current_price
                ))

            # Channel levels
            if 'lines' in results and middleware_name == 'channels':
                levels.extend(self._extract_channel_levels(
                    results['lines'], timeframe, current_price
                ))

            # Pivot levels from zigzag
            if 'pivots' in results and middleware_name == 'zigzag':
                levels.extend(self._extract_pivot_levels(
                    results['pivots'], timeframe, current_price
                ))

        return levels

    def _extract_volume_profile_levels(self, lines: List, timeframe: ChartInterval,
                                       current_price: float) -> List[LevelInfo]:
        """Extract POC, VAH, VAL levels from volume profile data"""
        levels = []

        for line in lines:
            try:
                # Line structure: (pivot1, pivot2, line_type)
                if len(line) >= 3:
                    pivot1, pivot2, line_type = line

                    # Extract price from pivot (timestamp, price, type)
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        # Determine if support or resistance based on price relative to current
                        level_type = 'support' if price < current_price else 'resistance'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.8,  # Base strength for detected levels
                            distance=distance,
                            level_type=level_type,
                            timeframe=timeframe,
                            last_test_time=pivot1[0] if len(pivot1) >= 1 else None
                        ))

            except Exception as e:
                continue  # Skip invalid lines

        return levels

    def _extract_support_resistance_levels(self, lines: List, timeframe: ChartInterval,
                                           current_price: float) -> List[LevelInfo]:
        """Extract support/resistance levels from lines"""
        levels = []

        for line in lines:
            try:
                # Line structure: (pivot1, pivot2, line_type)
                if len(line) >= 3:
                    pivot1, pivot2, line_type = line

                    # Extract price from pivot (timestamp, price, type)
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        # Determine if support or resistance based on price relative to current
                        level_type = 'support' if price < current_price else 'resistance'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.8,  # Base strength for detected levels
                            distance=distance,
                            level_type=level_type,
                            timeframe=timeframe,
                            last_test_time=pivot1[0] if len(pivot1) >= 1 else None
                        ))

            except Exception as e:
                continue  # Skip invalid lines

        return levels

    def _extract_channel_levels(self, lines: List, timeframe: ChartInterval,
                                current_price: float) -> List[LevelInfo]:
        """Extract channel boundary levels"""
        levels = []

        for line in lines:
            try:
                if len(line) >= 3:
                    pivot1, pivot2, line_type = line
                    price = pivot1[1] if len(pivot1) >= 2 else None

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        # Channel type determines level type
                        if 'upper' in str(line_type):
                            level_type = 'channel_resistance'
                        elif 'lower' in str(line_type):
                            level_type = 'channel_support'
                        else:
                            level_type = 'channel_middle'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.7,  # Channel levels have good strength
                            distance=distance,
                            level_type=level_type,
                            timeframe=timeframe
                        ))

            except Exception as e:
                continue

        return levels

    def _extract_pivot_levels(self, pivots: List, timeframe: ChartInterval,
                              current_price: float) -> List[LevelInfo]:
        """Extract significant pivot levels"""
        levels = []

        # Only use recent pivots (last 20% of data)
        recent_pivots = pivots[-max(1, len(pivots) // 5):] if pivots else []

        for pivot in recent_pivots:
            try:
                if len(pivot) >= 3:
                    timestamp, price, pivot_type = pivot

                    if price and price > 0:
                        distance = abs(price - current_price) / current_price * 100

                        level_type = f'pivot_{pivot_type}' if pivot_type else 'pivot'

                        levels.append(LevelInfo(
                            price=price,
                            strength=0.6,  # Pivot levels have moderate strength
                            distance=distance,
                            level_type=level_type,
                            timeframe=timeframe,
                            last_test_time=timestamp
                        ))

            except Exception as e:
                continue

        return levels


class LevelBasedFeatureEngineer:
    """Creates features based on price interaction with key levels"""

    def __init__(self):
        self.proximity_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]  # Distance thresholds in %

    def create_level_features(self, current_price: float, current_volume: float,
                              levels: Dict[ChartInterval, List[LevelInfo]]) -> Dict[str, float]:
        """
        Create features based on current price interaction with levels

        Args:
            current_price: Current market price
            current_volume: Current volume
            levels: Dict of levels by timeframe

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # Flatten all levels
        all_levels = []
        for timeframe_levels in levels.values():
            all_levels.extend(timeframe_levels)

        if not all_levels:
            return self._get_default_features()

        # Sort levels by distance from current price
        all_levels.sort(key=lambda x: x.distance)

        # Proximity features
        features.update(self._create_proximity_features(current_price, all_levels))

        # Level strength features
        features.update(self._create_strength_features(all_levels))

        # Support/Resistance balance
        features.update(self._create_support_resistance_balance(current_price, all_levels))

        # Volume context
        features['current_volume'] = current_volume
        features['volume_normalized'] = min(current_volume / 1000000, 10.0)  # Normalize volume

        # Timeframe importance
        features.update(self._create_timeframe_features(levels))

        return features

    def _create_proximity_features(self, current_price: float,
                                   levels: List[LevelInfo]) -> Dict[str, float]:
        """Create features based on proximity to levels"""
        features = {}

        # Find closest levels in each direction
        closest_support = None
        closest_resistance = None

        for level in levels:
            if level.price < current_price and (not closest_support or level.distance < closest_support.distance):
                closest_support = level
            elif level.price > current_price and (not closest_resistance or level.distance < closest_resistance.distance):
                closest_resistance = level

        # Distance to closest levels
        features['distance_to_support'] = closest_support.distance if closest_support else 100.0
        features['distance_to_resistance'] = closest_resistance.distance if closest_resistance else 100.0

        # Support/resistance strength
        features['support_strength'] = closest_support.strength if closest_support else 0.0
        features['resistance_strength'] = closest_resistance.strength if closest_resistance else 0.0

        # Count levels within proximity thresholds
        for threshold in self.proximity_thresholds:
            count = sum(1 for level in levels if level.distance <= threshold)
            features[f'levels_within_{threshold}pct'] = count

        return features

    def _create_strength_features(self, levels: List[LevelInfo]) -> Dict[str, float]:
        """Create features based on level strength"""
        features = {}

        if not levels:
            return {'avg_level_strength': 0.0, 'max_level_strength': 0.0}

        strengths = [level.strength for level in levels]
        features['avg_level_strength'] = np.mean(strengths)
        features['max_level_strength'] = np.max(strengths)

        # Strength by level type
        level_types = {}
        for level in levels:
            if level.level_type not in level_types:
                level_types[level.level_type] = []
            level_types[level.level_type].append(level.strength)

        for level_type, type_strengths in level_types.items():
            features[f'{level_type}_strength'] = np.mean(type_strengths)

        return features

    def _create_support_resistance_balance(self, current_price: float,
                                           levels: List[LevelInfo]) -> Dict[str, float]:
        """Create features representing support/resistance balance"""
        features = {}

        support_levels = [l for l in levels if l.price < current_price]
        resistance_levels = [l for l in levels if l.price > current_price]

        features['support_count'] = len(support_levels)
        features['resistance_count'] = len(resistance_levels)
        features['support_resistance_ratio'] = (
            len(support_levels) / max(len(resistance_levels), 1)
        )

        # Weighted strength balance
        support_strength = sum(l.strength / max(l.distance, 0.1) for l in support_levels)
        resistance_strength = sum(l.strength / max(l.distance, 0.1) for l in resistance_levels)

        features['weighted_support_strength'] = support_strength
        features['weighted_resistance_strength'] = resistance_strength
        features['strength_balance'] = (
            support_strength / max(resistance_strength, 0.1)
        )

        return features

    def _create_timeframe_features(self, levels: Dict[ChartInterval, List[LevelInfo]]) -> Dict[str, float]:
        """Create features based on timeframe importance"""
        features = {}

        timeframe_weights = {'M': 3.0, 'W': 2.0, 'D': 1.0}  # Monthly > Weekly > Daily

        for timeframe, weight in timeframe_weights.items():
            if timeframe in levels:
                count = len(levels[timeframe])
                avg_strength = np.mean([l.strength for l in levels[timeframe]]) if levels[timeframe] else 0.0

                features[f'{timeframe}_level_count'] = count
                features[f'{timeframe}_avg_strength'] = avg_strength
                features[f'{timeframe}_weighted_importance'] = count * avg_strength * weight

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when no levels are available"""
        features = {}

        # Set all features to neutral/zero values
        features.update({
            'distance_to_support': 100.0,
            'distance_to_resistance': 100.0,
            'support_strength': 0.0,
            'resistance_strength': 0.0,
            'support_count': 0,
            'resistance_count': 0,
            'support_resistance_ratio': 1.0,
            'weighted_support_strength': 0.0,
            'weighted_resistance_strength': 0.0,
            'strength_balance': 1.0,
            'avg_level_strength': 0.0,
            'max_level_strength': 0.0,
            'current_volume': 1000000,
            'volume_normalized': 1.0
        })

        # Proximity features
        for threshold in self.proximity_thresholds:
            features[f'levels_within_{threshold}pct'] = 0

        # Timeframe features
        for timeframe in ['M', 'W', 'D']:
            features[f'{timeframe}_level_count'] = 0
            features[f'{timeframe}_avg_strength'] = 0.0
            features[f'{timeframe}_weighted_importance'] = 0.0

        return features


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
