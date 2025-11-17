# 📊 Interactive Environment Visualizer - Specification

## Overview
An interactive Jupyter notebook tool to step through the trading environment, visualizing what the agent "sees" at each step. This helps debug reward engineering and feature engineering by showing real-time decision-making context.

---

## Core Features

### 1. Price Chart with Context
- **Candlestick chart** showing the lookback window (what agent sees)
- **Volume Profile overlay**: VP bins, POC, VAH, VAL
- **Support/Resistance levels**: Previous day/week high/low
- **Naked POCs**: Untested volume profile POCs
- **Trade markers**: Entry/exit points, SL/TP levels
- **Current price indicator**: Highlighted current bar

### 2. Feature Activation Matrix
Visual heatmap showing which features are "active" at current step:

**Feature Groups**:
- 🟦 **Micro Temporal** (5 features): OHLC, Volume
- 🟧 **Micro Spatial** (4 features): Body/Wick ratios
- 🟩 **Meso Patterns** (2 features): 1h, 4h returns
- 🟪 **Macro Patterns** (1 feature): 24h return
- 🟨 **Account State** (5 features): Balance, Equity, PnL
- 🟥 **Position Info** (7 features): Position size, leverage, SL/TP
- 🟫 **VP Bins** (top 5 active bins shown)
- ⬜ **VP Levels** (9 features): Continuous + Binary

**Color Coding**:
- 🟢 **Green**: Positive/bullish (value > 0.3)
- 🟡 **Yellow**: Neutral (-0.3 to 0.3)
- 🔴 **Red**: Negative/bearish (value < -0.3)
- ⚪ **Gray**: Zero/inactive

### 3. Interactive Controls
- **◀ Previous**: Step backward
- **▶ Next**: Step forward (execute action)
- **⏸ Pause / ▶️ Play**: Auto-advance mode
- **Slider**: Jump to specific step
- **Action Override**: Manual action selection (HOLD/LONG/SHORT)
- **Speed Control**: Playback speed (0.1s - 5s per step)

### 4. State Information Panel
Real-time display of:
```
Step: 5432 / 10000
Action: LONG (Agent)
Reward: +0.234
Position: +1.5 BTC @ $45,234
Balance: $10,234 (+2.34%)
Equity: $10,567
Unrealized PnL: +$333 (+3.25%)
Trade Duration: 12 bars
Stop Loss: $44,500 (-1.62%)
Take Profit: $47,000 (+3.90%)
Action Mask: [HOLD ✓] [LONG ✗] [SHORT ✗]
```

### 5. Reward Breakdown Panel
Show reward components at each step:
```
Reward Breakdown:
  Bankruptcy Penalty: 0.000
  Trade Closed: 0.000
  Position Opened: +0.400
  Unrealized PnL: +0.015
  Hold Penalty: 0.000
  ─────────────────────
  Total Reward: +0.415
```

---

## Implementation Structure

### New Files

#### 1. `Interactive_Env_Visualizer.ipynb`
Main notebook with cells:
1. **Setup**: Import libraries, configuration
2. **Data Loading**: Load pickle, create environment
3. **Model Loading**: Load trained model (optional)
4. **Visualizer Launch**: Initialize and display interactive dashboard
5. **Analysis Tools**: Helper cells for debugging specific patterns

#### 2. `src/environments/interactive_visualizer.py`
Core visualization class:

```python
class InteractiveEnvVisualizer:
    def __init__(self, env, model=None, history_size=1000):
        """
        Args:
            env: SimpleTradingEnv instance
            model: Trained PPO model (optional, for action predictions)
            history_size: Max steps to keep in memory
        """
        
    def display(self):
        """Launch interactive dashboard with ipywidgets"""
        
    def step_forward(self, action=None):
        """Advance one step, use model if action=None"""
        
    def step_backward(self):
        """Go back one step (replay from history)"""
        
    def update_visualization(self):
        """Refresh all plots for current step"""
        
    def plot_price_chart(self):
        """Plot candlesticks + VP + levels"""
        
    def plot_feature_matrix(self, obs):
        """Plot feature activation heatmap"""
        
    def get_state_info(self):
        """Get formatted state information"""
        
    def get_reward_breakdown(self):
        """Break down reward into components"""
        
    def export_chart(self, filename):
        """Save current view as PNG"""
```

### Extended Files

#### `src/environments/simple_trading_env.py`
Add methods for visualization support:

```python
def get_reward_breakdown(self, action, current_state, previous_state):
    """
    Returns dict with reward components:
    {
        'bankruptcy': -1.0 or 0.0,
        'trade_closed': float,
        'position_opened': float,
        'unrealized_pnl': float,
        'hold_penalty': float,
        'total': float
    }
    """
```

#### `src/environments/generic_trading_visualizer.py`
Add new plotting methods:

```python
def plot_support_resistance_levels(fig, data, current_step):
    """Add support/resistance lines to chart"""
    
def plot_feature_highlights(fig, obs, current_step):
    """Highlight bars where features are active"""
```

---

## Chart Layout (Plotly Subplots)

```
┌─────────────────────────────────────────────┐
│  Price Chart (4 rows)                       │
│  - Candlesticks                             │
│  - Volume Profile (right side)              │
│  - POC/VAH/VAL lines                        │
│  - Support/Resistance zones                 │
│  - Trade markers (▲▼✕)                      │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Position Size (1 row)                      │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Equity Curve (1 row)                       │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Reward Signal (1 row)                      │
└─────────────────────────────────────────────┘
```

**Total Height**: 7 rows (4:1:1:1 ratio)

---

## Feature Matrix Layout (Matplotlib)

```
┌──────────────────────────────────────────────┐
│ Micro Temporal    │ ● ● ● ● ●                │
│ Micro Spatial     │ ● ● ● ●                  │
│ Meso Patterns     │ ● ●                      │
│ Macro Patterns    │ ●                        │
│ Account State     │ ● ● ● ● ●                │
│ Position Info     │ ● ● ● ● ● ● ●            │
│ VP Bins (Top 5)   │ ● ● ● ● ●                │
│ VP Levels         │ ● ● ● ● ● ● ● ● ●        │
└──────────────────────────────────────────────┘

Legend: 🟢 Bullish | 🟡 Neutral | 🔴 Bearish | ⚪ Inactive
```

---

## Widget Layout (ipywidgets)

```
┌────────────────────────────────────────────────┐
│  Controls                                      │
│  [◀ Prev] [▶ Next] [▶️ Play] [Speed: 1.0s ▼] │
│  [Step: ═════●══════ 5432/10000]              │
│  [Action: Agent ▼]                            │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  Price Chart (Plotly)                          │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  Feature Matrix (Matplotlib)                   │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  State Info Panel                              │
│  Step: 5432 | Action: LONG | Reward: +0.234   │
│  Position: +1.5 BTC @ $45,234                  │
│  Balance: $10,234 (+2.34%)                     │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  Reward Breakdown                              │
│  Position Opened: +0.400                       │
│  Unrealized PnL:  +0.015                       │
│  Total Reward:    +0.415                       │
└────────────────────────────────────────────────┘
```

---

## Usage Flow

### Basic Usage
```python
# 1. Setup environment
data = pd.read_pickle('data/binance-BTCUSDT-5m.pkl')
env = SimpleTradingEnv(data, lookback_window=288)
env.reset()

# 2. Optional: Load model
model = PPO.load('trading_bot')

# 3. Launch visualizer
viz = InteractiveEnvVisualizer(env, model=model)
viz.display()

# 4. Interact using buttons or programmatically
viz.step_forward()  # Advance one step
viz.step_backward()  # Go back
```

### Manual Testing Mode
```python
# Override agent actions manually
viz = InteractiveEnvVisualizer(env)  # No model
viz.display()

# User selects actions via dropdown
# Test reward function by trying different actions at key moments
```

### Analysis Mode
```python
# Record interesting moments
interesting_steps = []

def on_step_change(step_idx):
    reward = viz.get_current_reward()
    if abs(reward) > 0.1:  # High reward/penalty
        interesting_steps.append(step_idx)

viz.on_step_change = on_step_change
viz.play()  # Auto-run through episode

# Review interesting steps
for step in interesting_steps:
    viz.jump_to_step(step)
    viz.export_chart(f"step_{step}.png")
```

---

## Implementation Phases

### Phase 1: MVP (Minimum Viable Product)
- ✅ Basic notebook structure
- ✅ Environment setup and reset
- ✅ Simple step forward/backward buttons
- ✅ Price chart with candlesticks only
- ✅ State info panel (text)

**Goal**: Working prototype to step through environment

### Phase 2: Feature Visualization
- ✅ Extract observations at each step
- ✅ Feature activation matrix (basic heatmap)
- ✅ Color coding for feature values
- ✅ Legend for feature groups

**Goal**: See which features are active

### Phase 3: Advanced Chart Features
- ✅ Volume Profile overlay (bins, POC, VAH, VAL)
- ✅ Support/Resistance levels
- ✅ Trade markers (entry/exit/SL/TP)
- ✅ Highlight current bar

**Goal**: Complete trading context visualization

### Phase 4: Interactive Enhancements
- ✅ Play/Pause mode with speed control
- ✅ Action override dropdown
- ✅ Reward breakdown panel
- ✅ Export chart button

**Goal**: Full interactive debugging tool

### Phase 5: Advanced Analysis (Future)
- ❓ Compare multiple models side-by-side
- ❓ Heatmap of "good entry zones" based on historical performance
- ❓ Feature importance visualization (SHAP values)
- ❓ Pattern detection (mark similar setups)

---

## Key Benefits

### 1. Reward Function Debugging
- See exactly which actions trigger rewards/penalties
- Identify if rewards align with good trading decisions
- Test edge cases (near support, after big move, etc.)

### 2. Feature Engineering Validation
- Check if features actually capture intended patterns
- Identify redundant features (always same color)
- Spot missing features (important patterns not captured)

### 3. Model Behavior Understanding
- See why agent chooses certain actions
- Identify bias (always LONG, never trades near support, etc.)
- Validate action mask logic

### 4. Training Data Quality Check
- Spot data issues (gaps, outliers, incorrect VP calculations)
- Verify lookback window shows relevant context
- Check feature normalization

---

## Dependencies

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Interactive widgets
import ipywidgets as widgets
from IPython.display import display, clear_output

# RL components
from stable_baselines3 import PPO
from environments.simple_trading_env import SimpleTradingEnv

# Optional
import torch  # For model debugging
```

---

## Configuration Options

```python
CONFIG = {
    'lookback_window': 288,           # Bars to show in chart
    'chart_height': 800,              # Total chart height (px)
    'feature_matrix_width': 12,       # Width in inches
    'feature_matrix_height': 6,       # Height in inches
    'play_speed_default': 1.0,        # Seconds per step
    'play_speed_range': (0.1, 5.0),   # Min/max speed
    'history_size': 1000,             # Max steps to keep
    'color_scheme': {
        'bullish': '#00ff00',         # Green
        'neutral': '#ffff00',         # Yellow
        'bearish': '#ff0000',         # Red
        'inactive': '#cccccc',        # Gray
    },
    'feature_thresholds': {
        'bullish': 0.3,               # Value > 0.3 = green
        'bearish': -0.3,              # Value < -0.3 = red
    }
}
```

---

## Questions & Decisions

### ✅ Confirmed
- Use **Plotly** for price chart (interactive zoom/pan)
- Use **Matplotlib** for feature matrix (simpler for heatmaps)
- Show **last timestep** of features (current state)
- Include **action override** for manual testing
- Add **reward breakdown** panel

### ❓ To Decide
1. **Feature matrix detail**: Show all feature values or just binary active/inactive?
   - **Recommendation**: Show actual values with tooltip on hover

2. **Support/Resistance**: Auto-detect or use fixed levels from data?
   - **Recommendation**: Use levels already in environment (prev day/week, naked POCs)

3. **Trade statistics**: Add cumulative stats panel (win rate, total PnL)?
   - **Recommendation**: Phase 2 - focus on step-by-step first

4. **Save functionality**: Save entire session or just current chart?
   - **Recommendation**: Just chart for now (PNG export)

---

## Next Steps

1. ✅ **Create notebook skeleton** (`Interactive_Env_Visualizer.ipynb`)
2. ✅ **Implement Phase 1 (MVP)**: Basic stepping and price chart
3. ⏳ **Test with real environment**: Verify it works with existing setup
4. ⏳ **Iterate based on feedback**: Add features as needed

**Estimated Time**: 
- Phase 1: ~1 hour
- Phase 2: ~1 hour
- Phase 3: ~2 hours
- Phase 4: ~1 hour

**Total**: ~5 hours for full implementation

---

## Success Criteria

✅ **MVP Success**:
- Can step forward/backward through environment
- See candlestick chart for current window
- Read current state (position, balance, reward)

✅ **Full Feature Success**:
- Feature matrix shows all observation components
- Can identify when/why agent takes actions
- Reward breakdown helps debug reward function
- Can manually test actions and see results

✅ **Production Ready**:
- Stable (no crashes on edge cases)
- Fast (<1 second to update visualization)
- Intuitive UI (easy for non-technical users)
- Well-documented with examples

---

**Ready to implement? Let me know if anything needs adjustment!** 🚀
