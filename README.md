# 🤖 RL Cryptocurrency Trading Bot

A reinforcement learning-powered trading system for cryptocurrency markets using PPO (Proximal Policy Optimization).

## 🎯 What It Does

- **Learns to Trade**: Uses reinforcement learning to make buy/sell decisions
- **Multi-Symbol Support**: Train on different cryptocurrencies (BTCUSDT, ETHUSDT, etc.)
- **Feature Engineering**: Combines technical indicators across multiple timeframes
- **Performance Tracking**: Comprehensive trading metrics and reporting

## 🏗️ Core Components

### **Data Pipeline**
- Loads OHLCV data from JSON files
- Adds technical indicators (RSI, MACD, moving averages, etc.)
- Interpolates higher timeframe data to lower timeframes
- Caches processed features for faster training

### **RL Environment**
- **Dual Action Space**: Signal [-1, 1] + Position Size [0.001, 1] for sophisticated trading control
- **Position Management**: Smart position scaling/reduction instead of always closing positions
- **Action Memory**: Model receives feedback about its previous decisions to learn consistency
- **Realistic Trading**: PnL tracking, unrealized gains, position exposure, and trade execution
- **Enhanced Metrics**: Win rates, drawdown, trade frequency, and portfolio performance
- **Configurable Rewards**: Customizable reward functions for different trading strategies

### **Trading Intelligence**
The environment supports sophisticated trading strategies:

**Position Scaling Example:**
- Current: 30% long position
- Model Action: `[0.8, 0.8]` (strong buy signal, 80% position size)
- Result: Scales up to 80% long position (adds 50%)

**Position Reduction Example:**
- Current: 80% long position  
- Model Action: `[0.5, 0.3]` (weak buy signal, 30% position size)
- Result: Reduces to 30% long position (sells 50%)

**Direction Change Example:**
- Current: 50% long position
- Model Action: `[-0.6, 0.4]` (sell signal, 40% position size)  
- Result: Closes long and opens 40% short position

### **Model Training**
- PPO algorithm with GPU support
- Early stopping and progress tracking
- Comprehensive training reports
- Model persistence and loading

### **Prediction & Analysis**
- Generate trading signals from trained models
- Real-time portfolio tracking
- Detailed performance analytics
- Backtesting capabilities

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
```python
from src.prediction.rl_predictor import RLPredictor
from src.training.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_data('BTCUSDT')

# Train model with dual action space (signal + position size)
predictor = RLPredictor()
predictor.train(data['15m'])
```

### 3. Understand Model Actions
The trained model outputs two values:
- **Signal [-1, 1]**: Trading direction (sell/hold/buy)
- **Position Size [0.001, 1]**: Fraction of account to use

```python
# Example model outputs:
# [0.8, 0.6] = Strong buy signal, use 60% of account
# [-0.5, 0.3] = Moderate sell signal, use 30% of account  
# [0.05, 0.1] = Weak signal, model chooses to hold (below threshold)
```

### 4. Generate Predictions
```python
# Load trained model and predict
predictor.load_model()
predictions = predictor.generate_predictions(test_data)

# Predictions contain both signal and position size information
# for sophisticated position management
```

## 📁 Key Files

- `src/prediction/rl_predictor.py` - Main RL training and prediction
- `src/environments/trading_environment.py` - RL trading simulation
- `src/training/data_loader.py` - Data loading and feature engineering
- `src/core/normalization_config.py` - Feature configuration
- `src/reporting/model_training_report.py` - Training analysis

## � Training Notebooks

- `RL_Trading_Model_Training.ipynb` - Interactive model training
- `VectorBT_Backtest.ipynb` - Performance backtesting
- `Candlestick_Chart_Visualizer.ipynb` - Data visualization

## �️ Configuration

The system is configurable through:
- Feature selection and normalization
- Environment parameters (reward functions, window sizes)
- Training hyperparameters (learning rates, batch sizes)
- Symbol selection and timeframes

## � Output

Training produces:
- Trained PPO model files
- Comprehensive training reports
- Trading performance metrics
- Backtesting results and visualizations

---

*A machine learning approach to cryptocurrency trading using modern reinforcement learning techniques.*