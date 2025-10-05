"""
🎯 AUTONOMOUS TRADING SYSTEM - Usage Guide & Examples
===================================================
This guide demonstrates how to use the new autonomous trading system that learns
to make trading decisions based on multi-timeframe technical analysis.

Key Features:
- Multi-timeframe level extraction (M, W, D)
- Level-based feature engineering
- Autonomous trading decisions (buy/sell/hold)
- Real-time market data processing
- Backtesting capabilities

System Architecture:
1. MultitimeframeLevelExtractor - Extracts key levels from higher timeframes
2. LevelBasedFeatureEngineer - Creates features based on price-level interactions
3. AutonomousTrader - Makes trading decisions using ML models
4. DataPipelineManager - Manages data flow and real-time processing
5. AutonomousTraderTrainer - Trains models on historical data
"""

from src.autonomous_trader import test_autonomous_trader
from src.data_pipeline import DataPipelineManager


def quick_start_guide():
    """Quick start guide for the autonomous trading system"""
    print("🚀 AUTONOMOUS TRADING SYSTEM - Quick Start Guide")
    print("=" * 60)

    print("\n📋 Step 1: Test Level Extraction")
    print("-" * 30)
    print("The system extracts key levels from multiple timeframes:")
    test_autonomous_trader()

    print("\n📋 Step 2: Initialize Data Pipeline")
    print("-" * 30)
    print("Set up the data pipeline for real-time processing:")

    pipeline = DataPipelineManager()
    success = pipeline.initialize_pipeline('BTCUSDT', load_models=False)

    if success:
        print("✅ Pipeline initialized successfully")

        # Show extracted levels
        status = pipeline.get_pipeline_status()
        btc_summary = status['levels_summary']['BTCUSDT']
        print(f"📊 Extracted {btc_summary['total_levels']} levels from {len(btc_summary['timeframes'])} timeframes")

        # Test signal generation
        sample_data = {
            'symbol': 'BTCUSDT',
            'close': 65000.0,
            'volume': 2000000,
            'high': 65200.0,
            'low': 64800.0,
            'open': 64950.0
        }

        signal = pipeline.get_trading_signal(sample_data)
        print(f"🎯 Generated trading signal: {signal.action.value} (confidence: {signal.confidence:.1%})")

    print("\n📋 Step 3: Train Models (Optional)")
    print("-" * 30)
    print("To train autonomous trading models, run:")
    print("python -c \"from src.autonomous_trader_trainer import train_autonomous_trading_system; train_autonomous_trading_system()\"")
    print("Note: This requires sufficient historical data and may take time")

    print("\n📋 Step 4: Real-time Usage")
    print("-" * 30)
    print("For real-time trading decisions:")

    # Example of real-time usage
    real_time_example = """
# Initialize pipeline with trained models
pipeline = DataPipelineManager()
pipeline.initialize_pipeline('BTCUSDT', load_models=True)

# Start background level updates
pipeline.start_background_updates()

# Process incoming market data
while True:
    market_data = get_current_market_data()  # Your data source
    signal = pipeline.get_trading_signal(market_data)
    
    if signal.confidence > 0.7:  # High confidence threshold
        execute_trade(signal)  # Your trading execution logic
"""

    print(real_time_example)

    print("\n✅ Quick start guide completed!")


def feature_showcase():
    """Showcase the key features of the system"""
    print("\n🌟 FEATURE SHOWCASE")
    print("=" * 60)

    print("\n1. 📊 Multi-Timeframe Level Extraction")
    print("   - Monthly (M): Long-term structural levels")
    print("   - Weekly (W): Medium-term trend levels")
    print("   - Daily (D): Short-term support/resistance")
    print("   - Extracts: POC, VAH, VAL, channels, pivots, S/R levels")

    print("\n2. 🧠 Level-Based Feature Engineering")
    print("   - Distance to nearest support/resistance")
    print("   - Level strength and test frequency")
    print("   - Support/resistance balance")
    print("   - Volume profile interactions")
    print("   - Timeframe importance weighting")

    print("\n3. 🤖 Autonomous Trading Decisions")
    print("   - BUY: When price approaches strong support")
    print("   - SELL: When price approaches strong resistance")
    print("   - HOLD: When no clear opportunity exists")
    print("   - Includes entry, stop-loss, take-profit levels")

    print("\n4. 🔄 Real-Time Processing")
    print("   - Background level updates")
    print("   - Market data processing")
    print("   - Signal generation with confidence scores")
    print("   - Risk management integration")

    print("\n5. 📈 Backtesting & Validation")
    print("   - Historical simulation")
    print("   - Performance metrics")
    print("   - Model validation")
    print("   - Feature importance analysis")


def architecture_overview():
    """Explain the system architecture"""
    print("\n🏗️ SYSTEM ARCHITECTURE")
    print("=" * 60)

    architecture = """
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   M/W/D Data    │───▶│ Level Extractor │───▶│  Key Levels     │
    │   (Higher TF)   │    │                 │    │  (S/R, POC,     │
    │                 │    │                 │    │   Channels)     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                            │
                                                            ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Intraday Data  │───▶│ Feature Engineer│◄───│   Combined      │
    │   (1h, 4h)      │    │                 │    │   Features      │
    │                 │    │                 │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                            │
                                                            ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Trading Signal  │◄───│ Autonomous      │◄───│  ML Models      │
    │ (Buy/Sell/Hold) │    │ Trader          │    │ (RandomForest,  │
    │                 │    │                 │    │  GradBoost)     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
    """

    print(architecture)

    print("\nComponent Details:")
    print("• MultitimeframeLevelExtractor: Processes M/W/D data to find key levels")
    print("• LevelBasedFeatureEngineer: Creates features from price-level interactions")
    print("• AutonomousTrader: Uses ML models to make trading decisions")
    print("• DataPipelineManager: Orchestrates real-time data processing")
    print("• AutonomousTraderTrainer: Trains models on historical data")


def usage_examples():
    """Provide detailed usage examples"""
    print("\n💡 USAGE EXAMPLES")
    print("=" * 60)

    print("\n1. Basic Level Extraction:")
    example1 = '''
from src.autonomous_trader import MultitimeframeLevelExtractor

extractor = MultitimeframeLevelExtractor()
data_files = {
    'M': 'data/BTCUSDT-M.json',
    'W': 'data/BTCUSDT-W.json',
    'D': 'data/BTCUSDT-D.json'
}

levels = extractor.extract_levels_from_data(data_files)
print(f"Extracted {sum(len(lvls) for lvls in levels.values())} levels")
'''
    print(example1)

    print("\n2. Real-time Trading Pipeline:")
    example2 = '''
from src.data_pipeline import DataPipelineManager

# Initialize pipeline
pipeline = DataPipelineManager()
pipeline.initialize_pipeline('BTCUSDT', load_models=True)

# Process market data
market_data = {
    'symbol': 'BTCUSDT',
    'close': 65000.0,
    'volume': 2000000,
    'high': 65200.0,
    'low': 64800.0
}

signal = pipeline.get_trading_signal(market_data)
print(f"Action: {signal.action.value}")
print(f"Confidence: {signal.confidence:.1%}")
print(f"Reasoning: {signal.reasoning}")
'''
    print(example2)

    print("\n3. Model Training:")
    example3 = '''
from src.autonomous_trader_trainer import AutonomousTraderTrainer

trainer = AutonomousTraderTrainer()

# Prepare training data
features_df, labels_dict = trainer.prepare_training_data(
    'data/BTCUSDT-1h.json',  # Intraday trading data
    {
        'M': 'data/BTCUSDT-M.json',
        'W': 'data/BTCUSDT-W.json',
        'D': 'data/BTCUSDT-D.json'
    }
)

# Train models
results = trainer.train_models(features_df, labels_dict)
trainer.save_models()
'''
    print(example3)


def best_practices():
    """Share best practices for using the system"""
    print("\n📚 BEST PRACTICES")
    print("=" * 60)

    practices = [
        "🎯 Data Quality: Ensure clean, complete data across all timeframes",
        "⏰ Update Frequency: Update levels regularly (M: weekly, W: daily, D: 4-hourly)",
        "📊 Confidence Thresholds: Use confidence > 0.7 for high-probability trades",
        "🛡️ Risk Management: Always use stop-losses and position sizing",
        "📈 Backtesting: Validate models on out-of-sample data before live trading",
        "🔄 Model Retraining: Retrain models periodically as market conditions change",
        "📱 Monitoring: Monitor system performance and level quality regularly",
        "🎚️ Parameter Tuning: Adjust profit/loss thresholds based on market volatility"
    ]

    for practice in practices:
        print(f"   {practice}")

    print(f"\n⚠️ Important Notes:")
    print("   • This system is for educational/research purposes")
    print("   • Always paper trade before using real money")
    print("   • Past performance doesn't guarantee future results")
    print("   • Consider transaction costs and slippage in real trading")


def troubleshooting():
    """Common issues and solutions"""
    print("\n🔧 TROUBLESHOOTING")
    print("=" * 60)

    issues = [
        {
            "issue": "No levels extracted",
            "solution": "Check data file paths and ensure files contain sufficient data"
        },
        {
            "issue": "Low model confidence",
            "solution": "Retrain with more data or adjust profit/loss thresholds"
        },
        {
            "issue": "Models not loading",
            "solution": "Run the trainer first to create model files"
        },
        {
            "issue": "JSON serialization errors",
            "solution": "Check for datetime objects in export data"
        },
        {
            "issue": "Memory issues with large datasets",
            "solution": "Process data in chunks or reduce lookback periods"
        }
    ]

    for i, item in enumerate(issues, 1):
        print(f"{i}. ❌ Issue: {item['issue']}")
        print(f"   ✅ Solution: {item['solution']}\n")


def main():
    """Run the complete guide"""
    quick_start_guide()
    feature_showcase()
    architecture_overview()
    usage_examples()
    best_practices()
    troubleshooting()

    print("\n🎉 CONGRATULATIONS!")
    print("=" * 60)
    print("You now have a complete autonomous trading system that:")
    print("✅ Extracts key levels from multiple timeframes")
    print("✅ Engineers features based on price-level interactions")
    print("✅ Makes autonomous trading decisions using ML")
    print("✅ Processes real-time market data")
    print("✅ Provides backtesting and validation capabilities")
    print("\nNext steps:")
    print("1. Train models on your historical data")
    print("2. Paper trade to validate performance")
    print("3. Gradually transition to live trading with proper risk management")
    print("\nHappy trading! 🚀📈")


if __name__ == "__main__":
    main()
