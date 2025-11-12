# Trade Logging & Reward Shaping System

## Overview

This system allows you to:
1. **Log all trades** during training with full VP context
2. **Visualize and label** trades interactively (good/bad entries)
3. **Use labeled trades** to shape rewards and guide future training

This creates a feedback loop where you teach the model what "good" VP-aware entries look like.

---

## Architecture

### 1. TradeLogger (`src/environments/trade_logger.py`)
- Saves trades to PKL on environment reset
- Lightweight: just handles file I/O
- Automatically manages file rotation (max 10k trades per file)

### 2. SimpleBroker Integration
- **Broker already tracks all trades** in `trade_history`
- Environment enriches broker trades with VP context on `reset()`
- No duplicate tracking needed!

### 3. TradeVisualizer (`trade_visualizer.py`)
- Interactive matplotlib GUI
- Shows candlestick chart with VP levels (VAH, POC, VAL, Value Area)
- Entry/exit markers with P&L
- Buttons: "Good Entry", "Bad Entry", "Neutral", "Skip"
- Saves labeled trades to new PKL file

### 3. TradeRewardShaper (`src/environments/trade_reward_shaper.py`)
- Loads labeled trades
- Calculates average VP patterns for good/bad entries
- Returns bonus/penalty (+0.5 to -0.5) based on similarity to labeled patterns

### 4. Integration (`SimpleTradingEnv`)
- Uses broker's existing `trade_history` (no duplicate tracking)
- On `reset()`: enriches trades with VP context, logs to PKL
- Simple and efficient!

---

## Workflow

### Step 1: Train with Logging Enabled

```python
from environments.simple_trading_env import SimpleTradingEnv

# Enable logging (only on env 0 to avoid duplicates in vectorized training)
env = SimpleTradingEnv(
    data=train_data,
    initial_balance=10000,
    lookback_window=288,
    enable_trade_logging=True,      # Enable logging
    trade_log_dir="logs/trades"     # Output directory
)

# Train normally - trades will be saved automatically
model.learn(total_timesteps=100_000)
```

**Output**: `logs/trades/trades_TIMESTAMP.pkl` with all trades

---

### Step 2: Label Trades Interactively

```bash
source .venv/Scripts/activate
python trade_visualizer.py \\
  --trades logs/trades/trades_20251112_143022.pkl \\
  --output logs/trades/trades_labeled.pkl
```

**Interactive GUI**:
- Shows candlestick chart with VP levels
- Entry/exit markers
- Click buttons to label:
  - **Good Entry**: Bought near VAL/POC support, sold near VAH/POC resistance
  - **Bad Entry**: Bought at VAH resistance, sold at VAL support
  - **Neutral**: No clear VP context
  - **Skip**: Ignore this trade

**Labeling Guidelines**:
- **Good LONG entries**: Near VAL (support), below POC, in Value Area
- **Good SHORT entries**: Near VAH (resistance), above POC
- **Bad LONG entries**: Near VAH (resistance), way above POC
- **Bad SHORT entries**: Near VAL (support), below POC

**Output**: `logs/trades/trades_labeled.pkl`

---

### Step 3: Use Labeled Trades for Reward Shaping

#### Option A: Add as Reward Component (Recommended)

Edit `simple_trading_env.py`, in `__init__`:

```python
from environments.trade_reward_shaper import TradeRewardShaper

# In __init__:
self.trade_reward_shaper = TradeRewardShaper('logs/trades/trades_labeled.pkl')
```

In `calculate_reward`, add entry quality component:

```python
def calculate_reward(self, action, current_state, previous_state):
    # ... existing components ...
    
    # === NEW: Entry Quality Component ===
    entry_quality_component = 0.0
    
    # Check if position just opened
    if current_state['position_type'] != 'FLAT' and previous_state['position_type'] == 'FLAT':
        action_name = 'LONG' if current_state['position_type'] == 'LONG' else 'SHORT'
        
        # Get current VP context
        vp_levels = get_vp_levels_features_visible(
            self.data, self.current_step, self.lookback_window, self.n_bins
        )
        last_vp = vp_levels[-1]  # Last timestep
        
        vp_context = {
            'dist_to_vah': float(last_vp[4]),     # close_to_vah
            'dist_to_poc': float(last_vp[8]),     # close_to_poc
            'dist_to_val': float(last_vp[12]),    # close_to_val
            'close_in_va': bool(last_vp[20] > 0.5),
            'close_above_poc': bool(last_vp[24] > 0.5),
        }
        
        # Get reward based on labeled patterns
        entry_quality_component = self.trade_reward_shaper.calculate_entry_quality_reward(
            action_name, vp_context
        )
    
    # Update weights
    weights = {
        'pnl': 0.55,              # Primary: PnL changes
        'close': 0.20,            # Secondary: trade outcomes
        'bankruptcy': 0.05,       # Tertiary: survival
        'hold_penalty': 0.05,     # Tertiary: avoid flatline
        'performance': 0.05,      # Tertiary: quality
        'validity': 0.05,         # Tertiary: valid actions
        'entry_quality': 0.05,    # NEW: VP-aware entry quality
    }
    
    reward = (
        weights['pnl'] * pnl_component +
        weights['close'] * close_component +
        weights['bankruptcy'] * bankruptcy_component +
        weights['hold_penalty'] * hold_penalty +
        weights['performance'] * performance_component +
        weights['validity'] * validity_component +
        weights['entry_quality'] * entry_quality_component  # NEW
    )
    
    return reward
```

---

## Expected Results

### Before Labeling
- Model may trade randomly without VP awareness
- Buys at resistance, sells at support (poor timing)

### After Labeling + Retraining
- Model learns from your labels
- **Good LONG entries**: Near VAL (support), below POC
  - Entry quality reward: +0.3 to +0.5
- **Good SHORT entries**: Near VAH (resistance), above POC
  - Entry quality reward: +0.3 to +0.5
- **Bad entries** get penalized: -0.3 to -0.5

### Iterative Improvement
1. Train with logging → collect 1000 trades
2. Label 100-200 representative trades
3. Retrain with reward shaping
4. Repeat: collect new trades, label edge cases, refine

---

## Files Created

1. **`src/environments/trade_logger.py`** - Trade logging infrastructure
2. **`trade_visualizer.py`** - Interactive labeling GUI
3. **`src/environments/trade_reward_shaper.py`** - Reward shaping from labels
4. **`example_trade_logging.py`** - Example training script

---

## Quick Start

```bash
# 1. Train with logging
source .venv/Scripts/activate
python example_trade_logging.py

# 2. Label trades
python trade_visualizer.py --trades logs/trades/trades_TIMESTAMP.pkl

# 3. Integrate reward shaping (edit simple_trading_env.py as shown above)

# 4. Retrain model
# ... your normal training script ...
```

---

## Troubleshooting

### No trades logged
- Check `enable_trade_logging=True` in env
- Check `logs/trades/` directory exists
- Only env 0 logs in vectorized training (prevents duplicates)

### Visualizer not showing
- Ensure matplotlib backend is working: `import matplotlib; matplotlib.use('TkAgg')`
- Check data path and trades path are correct

### Reward shaping not working
- Verify labeled trades file exists
- Check that trade_reward_shaper is initialized in `__init__`
- Print entry_quality_component to debug values

---

## Advanced: Custom Labeling Criteria

You can extend the visualizer to label based on:
- Outcome: Did trade profit? (add trade P&L to criteria)
- Timing: How long until profitable? (add duration analysis)
- Market regime: Trending vs ranging (add volatility context)

Example custom analyzer in `trade_visualizer.py`:

```python
def analyze_outcome_based_labels(trades):
    """Auto-label based on actual P&L outcomes."""
    for trade in trades:
        if trade['pnl_pct'] > 2.0:  # >2% profit
            # Check if entry was at good VP level
            if trade['action'] == 'LONG' and trade['dist_to_val'] < 0.1:
                trade['label'] = 'good_entry'
                trade['label_reason'] = 'Profitable LONG near VAL support'
            elif trade['action'] == 'SHORT' and trade['dist_to_vah'] < 0.1:
                trade['label'] = 'good_entry'
                trade['label_reason'] = 'Profitable SHORT near VAH resistance'
        elif trade['pnl_pct'] < -1.0:  # >1% loss
            trade['label'] = 'bad_entry'
            trade['label_reason'] = f'Losing {trade["action"]} entry'
```

---

## Next Steps

1. ✅ System implemented and integrated
2. ⏳ Train model with logging for 50k-100k steps
3. ⏳ Label 100-200 trades
4. ⏳ Add entry_quality component to reward
5. ⏳ Retrain and compare results

This creates a human-in-the-loop learning system where you teach the model your trading intuition about VP levels!
