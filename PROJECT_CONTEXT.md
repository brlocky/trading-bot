# Trading Bot - Project Context

## 📋 Project Overview
ML-powered cryptocurrency trading bot with GPU-accelerated XGBoost training and multi-timeframe technical analysis.

## 🏗️ Architecture

### Project Structure
```
trading-bot/
├── src/
│   ├── core/           # Trading types (TradingAction, TradingSignal, LevelInfo, TradeRecord)
│   ├── memory/         # TradeMemoryManager (1000 trades, performance tracking)
│   ├── detection/      # BounceDetector (support/resistance analysis)
│   ├── extraction/     # MultitimeframeLevelExtractor, LevelBasedFeatureEngineer
│   ├── backtesting/    # VectorBTBacktester, FeatureManager (optimized backtesting) ⭐ NEW
│   ├── trading/        # AutonomousTrader (main decision system)
│   ├── training/       # SimpleModelTrainer, AutonomousTraderTrainer
│   ├── prediction/     # SimpleModelPredictor, SimpleModelReporter
│   └── ta/             # TechnicalAnalysisProcessor, middlewares (zigzag, volume_profile, channels, levels)
├── data/               # Training data (BTCUSDT M/W/D/1h/15m)
├── data_test/          # Test data
├── models/             # Saved models (.joblib)
├── backtest_results/   # VectorBT backtest outputs (signals, trades, metrics) ⭐ NEW
├── feature_cache/      # Cached TA and level features (FeatureManager) ⭐ NEW
└── notebooks/          # Simple_Model_Debug.ipynb, VectorBT_Backtest.ipynb ⭐ UPDATED
```

## 🔑 Key Components

### 1. SimpleModelTrainer (src/training/model_trainer.py)
- **GPU-only XGBoost** training (tree_method='gpu_hist', 8 threads)
- Raises error if no GPU available (no CPU fallback)
- Memory features: 10 additional features from trade history
- Bounce detection: Identifies support/resistance bounces
- Training thresholds: profit=3.0%, loss=-2.0%

### 2. Data Pipeline
- JSON candlestick data → DataFrame conversion
- Multi-timeframe: M (monthly), W (weekly), D (daily), 1h, 15m
- Feature engineering: 50+ features from price-level interactions
- Labels: BUY/SELL/HOLD based on future price movements

### 3. Technical Analysis
- Custom TA processor with middleware pattern
- Middlewares: zigzag, volume_profile, channels, levels
- Support/resistance level extraction
- Volume profile analysis

## 🎯 Recent Changes (Oct 2025)

### Refactoring
- Extracted 3 monolithic files (2,496 lines) → 15 modular files
- Created proper package structure with `__init__.py` files
- Backward-compatible wrappers maintained

### Code Quality
- All linting issues fixed (Ruff)
- Renamed ambiguous variables (`l` → `level`)
- Fixed bare except statements (`except:` → `except Exception:`)
- Removed unused imports and f-strings

### Training Optimization
- **Simplified to XGBoost GPU only** (removed RandomForest, GradientBoosting, LogisticRegression)
- No CPU fallback - raises RuntimeError if GPU unavailable
- 5-fold cross-validation with F1 scoring

## 📝 Import Structure

### In Notebooks
```python
# Add src to path first
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Then import
from training.model_trainer import SimpleModelTrainer
from prediction.predictor import SimpleModelPredictor
from prediction.reporter import SimpleModelReporter
```

### Inside src/ Modules
```python
# Use relative imports
from .autonomous_trainer import AutonomousTraderTrainer
from ..memory.trade_memory import TradeMemoryManager
from core.trading_types import TradingAction
from ta.technical_analysis import TechnicalAnalysisProcessor
```

## 🚀 Current Workflow

### Training (Simple_Model_Debug.ipynb)
1. **Cell 1:** Title/Description
2. **Cell 2:** Core imports + path setup
3. **Cell 3:** Path setup (adds src to sys.path)
4. **Cell 4:** Training cell
   - Loads data (BTCUSDT M/W/D/1h/15m)
   - Configures trainer (profit=3%, loss=-2%)
   - Trains XGBoost GPU model
   - Saves to `models/simple_trading_model.joblib`
5. **Cell 5:** Load saved model
6. **Cell 6:** **VectorBT Backtesting** (1000 candles, vectorized) ⭐ UPDATED
   - Generates ML signals with thresholds
   - Runs VectorBT backtest with commissions/slippage
   - Displays performance metrics (Sharpe, drawdown, win rate)
   - Shows trade analysis

### VectorBT Backtesting (VectorBT_Backtest.ipynb) ⭐ NEW
- Comprehensive VectorBT workflow notebook
- Interactive visualizations (equity curve, drawdowns, trade markers)
- Export results to CSV
- Professional-grade performance metrics

### Model Configuration
```python
trainer = SimpleModelTrainer()
trainer.configure_training(
    profit_threshold=3.0,      # 3% profit target
    loss_threshold=-2.0,       # -2% stop loss
    lookforward_periods=[5, 10, 20]
)
```

## 🔧 Development Tools

### Linting
```bash
python -m ruff check src/
python -m ruff check --fix src/
```

### Testing
- Run notebook cells to test model training
- Live simulation for backtesting
- Plotly visualizations for results

## ⚠️ Important Notes

1. **GPU Required:** Training will fail without CUDA GPU
2. **Data Format:** JSON with `candles` array (time, open, high, low, close, volume)
3. **Memory System:** Tracks last 1000 trades, calculates win rate, bounce performance
4. **Thresholds:** BUY/SELL confidence must exceed 0.10 (10%) to generate signal

## 📊 Model Features

### Level-Based Features (50+)
- Support/resistance counts and distances
- Level strength and proximity
- Timeframe importance (M > W > D > 1h)
- Volume at levels

### Memory Features (10)
- Win rate (last 7 days)
- Average PnL
- Bounce win rate
- Consecutive wins/losses
- Market volatility regime

### Technical Indicators
- RSI, MACD, Bollinger Bands
- Volume ratio, volatility
- Price momentum

## 🚀 VectorBT Backtesting Integration ⭐

### Why VectorBT?
Replaced manual candle-by-candle simulation (250 candles) with professional vectorized backtesting:
- **10-20x faster** performance (1000 candles in <5 seconds after level loading)
- **Realistic execution modeling** (commissions, slippage, order fills)
- **Professional metrics** (Sharpe ratio, max drawdown, profit factor, win rate)
- **Interactive visualizations** (equity curves, trade markers, drawdown charts)
- **Trade analysis** (entry/exit prices, PnL per trade, holding periods)

### Performance Benchmarks
- **Old approach:** Manual simulation, 250 candles, ~30-60s (including level loading)
- **New approach:** VectorBT, 1000 candles, ~5s execution + 30-60s level loading (one-time)
- Level loading happens once per backtest regardless of candle count
- Vectorized operations enable processing 4x more candles in significantly less time

### VectorBTBacktester Class
Location: `src/backtesting/vectorbt_engine.py` (350+ lines)

**Key Methods:**
- `generate_signals()` - Converts XGBoost predictions (0-1 probabilities) to BUY/SELL/HOLD signals
- `run_backtest(freq='15T')` - Executes vectorized backtest with vbt.Portfolio.from_signals()
- `get_performance_stats()` - Returns dict with total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor
- `print_performance_summary()` - Formatted console output with all metrics
- `plot_results(use_widgets=False)` - Static Plotly charts (use_widgets=True for interactive widgets)
- `get_trade_analysis()` - Returns DataFrame of individual trades with entry/exit/PnL
- `export_results(output_dir)` - Saves signals.csv, trades.csv, performance_stats.txt, equity_curve.csv

### Configuration
```python
# Backtest parameters
initial_cash = 10000        # Starting capital
commission = 0.001          # 0.1% per trade
slippage = 0.0005           # 0.05% slippage
freq = '15T'                # 15-minute candles
buy_threshold = 0.10        # 10% minimum confidence for BUY
sell_threshold = 0.10       # 10% minimum confidence for SELL
```

### Usage Example
```python
# 1. Initialize backtester
from backtesting.vectorbt_engine import VectorBTBacktester
backtester = VectorBTBacktester(trainer, initial_cash=10000, commission=0.001, slippage=0.0005)

# 2. Generate ML signals from XGBoost predictions
signals_df = backtester.generate_signals(
    data=df_test,
    level_files=level_files,  # {M: path, W: path, D: path, 1h: path}
    buy_threshold=0.10,
    sell_threshold=0.10
)

# 3. Run vectorized backtest
portfolio = backtester.run_backtest(freq='15T')

# 4. View performance
backtester.print_performance_summary()
"""
Output:
==================================================
           VectorBT Backtest Results
==================================================
Total Return:        45.23%
Sharpe Ratio:        1.87
Max Drawdown:        -12.45%
Win Rate:            58.33%
Profit Factor:       2.14
Total Trades:        67
Avg Trade Duration:  4.3 hours
"""

# 5. Plot results
backtester.plot_results(use_widgets=False)  # Static plots

# 6. Export to CSV
backtester.export_results('backtest_results/')
```

### Integration with Notebooks
**Simple_Model_Debug.ipynb Cell 6** (lines 128-313):
- Replaced manual 250-candle simulation loop
- Now uses VectorBT with 1000 candles
- Generates signals → runs backtest → displays metrics → plots results

**VectorBT_Backtest.ipynb** (13 cells):
- Comprehensive demonstration notebook
- Shows all VectorBT features: setup, signal generation, backtesting, visualization, export
- Includes trade-by-trade analysis with entry/exit prices and PnL

### Technical Details
- **VectorBT version:** 0.28.1 (upgraded from 0.26.2 for Plotly 6.3.0 compatibility)
- **Signal format:** Boolean arrays for entries/exits passed to vbt.Portfolio.from_signals()
- **Level integration:** Uses existing MultitimeframeLevelExtractor and LevelBasedFeatureEngineer
- **Memory features:** TradeMemoryManager integration for historical trade performance
- **Plotting:** Static Plotly charts (use_widgets=False) to avoid anywidget dependency
- **Feature optimization:** FeatureManager with two-tier caching (8-10x faster feature calculation) ⭐ NEW

### FeatureManager System ⭐ NEW
**Two-Tier Feature Caching:**
- **Group 1 (Level Features)**: Updated once per day, uses MultitimeframeLevelExtractor
- **Group 2 (TA Features)**: Pre-calculated once for entire dataset, uses TechnicalAnalysisProcessor with middlewares
- **Configurable middlewares**: Custom middleware selection per timeframe
- **Performance**: 8-10x faster than candle-by-candle feature calculation
- **Cache storage**: `feature_cache/` directory with `.joblib` files
- **Smart invalidation**: TA cached by data range, levels cached by day

**Middleware Configuration Example:**
```python
custom_middleware_config = {
    '15m': [zigzag_middleware, levels_middleware],
    '1h': [zigzag_middleware, channels_middleware, levels_middleware],
    'D': [zigzag_middleware, channels_middleware, levels_middleware, volume_profile_middleware]
}

backtester = VectorBTBacktester(
    trainer=trainer,
    use_feature_manager=True,
    middleware_config=custom_middleware_config
)
```

### Future Enhancements
- **Walk-forward optimization** - Train on period N, test on N+1, retrain on N+N+1
- **Parameter optimization** - Use VectorBT's ParamIndexer for hyperparameter grid search
- **Position sizing** - Dynamic position sizing based on confidence levels
- **Risk management** - Trailing stops, time-based exits, max position limits
- **Multi-symbol backtesting** - Test portfolio across multiple cryptocurrencies
- **Backtest-to-training loop** - Use backtest results to retrain model with updated memory features

## 🎯 Next Steps / TODO
- [x] ✅ Replace manual simulation with VectorBT (Oct 2025)
- [x] ✅ Fix VectorBT/Plotly compatibility (upgraded to 0.28.1)
- [x] ✅ Add performance metrics and trade analysis
- [x] ✅ Implement FeatureManager with two-tier caching (Oct 2025)
- [ ] Test with different symbols (ADAUSDT, ETHUSDT, etc.)
- [ ] Implement walk-forward optimization
- [ ] Add VectorBT parameter grid search for hyperparameter tuning
- [ ] Dynamic position sizing based on ML confidence
- [ ] Risk management: trailing stops, max drawdown limits
- [ ] Multi-symbol portfolio backtesting
- [ ] Backtest-to-training feedback loop
- [ ] Incremental feature updates (only calculate new candles)
- [ ] Paper trading mode with live data feeds

## 📚 Dependencies
- **pandas, numpy** - Data manipulation
- **xgboost** (GPU version) - ML training with tree_method='gpu_hist'
- **scikit-learn** - Cross-validation, metrics
- **plotly** (6.3.0) - Visualizations
- **joblib** - Model serialization
- **vectorbt** (0.28.1) - Professional backtesting engine ⭐
- **ta-lib** (optional) - Technical indicators

## 🔗 Key Files to Reference
- `src/training/model_trainer.py` - Main training logic
- `src/prediction/predictor.py` - Prediction with thresholds
- `src/prediction/reporter.py` - Plotly visualization
- **`src/backtesting/vectorbt_engine.py`** - VectorBT backtesting engine ⭐
- **`src/backtesting/feature_manager.py`** - Optimized feature calculation with caching ⭐ NEW
- `Simple_Model_Debug.ipynb` - Training and backtesting workflow
- **`VectorBT_Backtest.ipynb`** - Comprehensive VectorBT examples with FeatureManager ⭐
- **`FEATURE_MANAGER_README.md`** - Feature system documentation ⭐ NEW
- `REFACTORING_SUMMARY.md` - Detailed refactoring notes
