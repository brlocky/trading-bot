# 🎨 Observation Monitor Guide

## Quick Start

### Option 1: Launch from Notebook
1. Open `RL_BACK.ipynb`
2. Run the "Launch Streamlit Observation Monitor" cell
3. Browser opens automatically at http://localhost:8501

### Option 2: Launch from Terminal
```bash
streamlit run streamlit_obs_monitor.py
```

---

## Features

### 📊 Heatmap Visualization
- **8 Feature Groups**: All observation components displayed as interactive heatmaps
- **Custom Color Scales**: Choose from RdBu, Viridis, Plasma, Hot, Cool, Portland, Picnic
- **Filter Groups**: Select which feature groups to display
- **Statistics**: Mean, std, min, max for each group
- **Distributions**: Optional histogram view

### 🎮 Interactive Controls

#### Manual Control
- **Reset Button**: Reset environment to initial state
- **Step Button**: Take one random action
- **Manual Action**: Choose specific action (HOLD/LONG/SHORT/CLOSE)

#### Auto Mode
- **Speed Control**: 1-10 steps per second
- **Auto Run**: Execute N steps automatically with progress tracking
- **Progress Bar**: Visual feedback during execution

### 📈 Statistics Tab
- **Summary Table**: All feature groups with stats
- **Correlation Matrix**: Group-level correlations
- **Value Ranges**: Check for NaN/Inf values

### 🎯 History Tab
- **Action Distribution**: Bar chart of actions taken
- **Cumulative Reward**: Line chart showing reward accumulation
- **Reward per Step**: Individual step rewards
- **Recent Steps**: Table of last 20 steps
- **Performance Summary**: Total steps, avg/best/worst rewards

---

## Configuration

### Sidebar Settings
- **Data Path**: Path to pickle file (default: `data/binance-BTCUSDT-5m.pkl`)
- **Lookback Window**: Number of timesteps (default: 288)
- **VP Bins**: Number of volume profile bins (default: 50)
- **Data Size**: Rows to load (1k - 50k)

### Display Options
- **Feature Groups**: Multi-select which groups to show
- **Color Scale**: Choose visualization colors
- **Show Statistics**: Toggle stats below heatmaps
- **Show Distributions**: Toggle histogram views

---

## Key Advantages

✅ **No Disk Writes** - All processing in-memory  
✅ **Real-Time Updates** - Instant visualization on actions  
✅ **Interactive** - Manual or automated stepping  
✅ **Comprehensive** - All 8 feature groups included  
✅ **Clean UI** - Organized tabs and filters  
✅ **Performance Tracking** - Action/reward history  

---

## Tips

1. **Start Small**: Use 5k-10k data size for fast loading
2. **Filter Groups**: Focus on specific features for detailed analysis
3. **Auto Mode**: Use for quick testing of observation changes
4. **Manual Actions**: Test specific trading scenarios
5. **Save Config**: Bookmark settings in browser for quick access

---

## Feature Groups Explained

### 1. Micro Temporal (5 features) - [288, 5]
- OHLC prices + Volume, normalized [0, 1]
- Shows price movement patterns over time

### 2. Micro Spatial (4 features) - [288, 4]
- Candle structure ratios [0, 1]
- Body, upper wick, lower wick, close position

### 3. Meso Patterns (2 features) - [288, 2]
- 1h and 4h returns [-1, 1]
- Short/medium term trends

### 4. Macro Patterns (1 feature) - [288, 1]
- 24h return [-1, 1]
- Long-term trend

### 5. Account State (5 features) - [288, 5]
- Balance, equity, PnL metrics [-1, 1]
- Account performance indicators

### 6. Position Info (7 features) - [288, 7]
- Position status, leverage, distances [-1, 1]
- Current trade information

### 7. VP Bins (50 features) - [288, 50]
- Volume distribution histogram [0, 1]
- Shows where volume traded across price levels

### 8. VP Levels (3 features) - [288, 3]
- VAH/VAL/POC distances [-1, 1]
- Key support/resistance levels

---

## Troubleshooting

### Port Already in Use
```bash
# Stop existing Streamlit process
streamlit kill
# Or use different port
streamlit run streamlit_obs_monitor.py --server.port 8502
```

### Browser Doesn't Open
Manually navigate to: http://localhost:8501

### Slow Performance
- Reduce data size (< 10k rows)
- Reduce lookback window (< 288)
- Disable distributions view
- Filter to fewer feature groups

### Missing Data File
Update `Data Path` in sidebar to correct pickle file location

---

## Next Steps

1. **Test with Different Data**: Load various timeframes/symbols
2. **Compare Observations**: Take screenshots before/after actions
3. **Track Patterns**: Watch how features evolve over time
4. **Debug Issues**: Use stats tab to check for NaN/Inf
5. **Optimize Features**: Identify which groups have most variance

---

**Enjoy real-time observation monitoring with zero disk writes!** 🚀
