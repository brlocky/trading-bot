# Expert Trade Replay System

## Overview

The Expert Trade Replay system implements curriculum learning by recording successful trades during training and replaying them to accelerate model learning.

## Architecture

### `ExpertTradeReplay` Class (`src/environments/expert_trade_replay.py`)

Standalone class that manages expert trade recording, storage, and injection.

**Key Features:**
- Records trades meeting success criteria (>2% profit, <50 steps duration)
- Persists trades to disk (`expert_trades.pkl`)
- Provides random trade selection for curriculum learning
- Manages memory with configurable max trades (default: 1000)
- Tracks statistics (avg PnL, duration, etc.)

### Integration with `SimpleTradingEnv`

The environment uses `ExpertTradeReplay` to:
1. **Record** successful trades during step()
2. **Inject** expert trades during reset() (20% probability)
3. **Persist** trades after training completes

## What Gets Recorded

Each expert trade contains:

```python
{
    'entry_step': 12500,              # Step index when trade opened
    'exit_step': 12535,               # Step index when trade closed
    'trade_object': {                 # Complete trade from broker
        'status': 'CLOSED',
        'entry_price': 45230.50,
        'exit_price': 46150.25,
        'tp_price': 46500.00,
        'sl_price': 44800.00,
        'position_size': 0.15,
        'pnl': 137.89,
        'pnl_percent': 2.45,
        'duration': 35,
        'commission': 12.50,
        'reason': 'TP',
        'risk_reward_ratio': 2.0,
        # ... more broker details
    },
    'actions': [                      # ALL actions during trade
        {'step': 12500, 'action': [1, 5, 3], 'close_price': 45230.50},  # OPEN LONG
        {'step': 12501, 'action': [0, 0, 0], 'close_price': 45245.20},  # HOLD
        {'step': 12502, 'action': [0, 0, 0], 'close_price': 45260.10},  # HOLD
        # ... all HOLD actions ...
        {'step': 12535, 'action': [3, 0, 0], 'close_price': 46150.25},  # CLOSE (or TP hit)
    ],
    'pnl_percent': 2.45,
    'hold_duration': 35,
    'timestamp': '2024-03-15 14:23:00'
}
```

## Trade Detection Logic

### Critical: Handles Position Reversals

Uses **direction** instead of **position_size** to detect trade closures:

```python
# WRONG (misses reversals):
if prev_position != 0 and curr_position == 0:
    # Only catches flat positions, not LONG→SHORT reversals

# CORRECT (catches everything):
if prev_direction != 0 and prev_direction != curr_direction:
    # Catches: LONG→FLAT, SHORT→FLAT, LONG→SHORT, SHORT→LONG
```

### Action Recording

Records **ALL actions** from trade open to close:
- Entry action (LONG/SHORT with ratio & ATR multiplier)
- All HOLD actions during position
- Exit action (CLOSE or automatic TP/SL)

## Curriculum Learning Flow

### Training Loop:

1. **Episode Reset**:
   - 20% chance: Inject expert trade setup
   - 80% chance: Normal random start
   
2. **During Episode**:
   - Record ALL actions while position is open
   - Detect trade closure (direction change detection)
   - Check success criteria (>2% profit, <50 steps)
   - If successful: Record to `ExpertTradeReplay`

3. **After Training**:
   - Collect expert trades from all parallel environments
   - Deduplicate by (entry_step, pnl_percent)
   - Save to `expert_trades.pkl`
   - Print statistics

### Expert Trade Injection:

When injecting an expert trade in reset():
```python
if expert_replay.should_inject(0.2):  # 20% chance
    expert = expert_replay.get_random_trade()
    start_step = expert['entry_step'] - lookback_window
    # Model sees the same setup that led to winning trade
    # Can compare model's actions vs expert's actions
```

## Benefits

1. **Faster Learning**: Model sees profitable patterns more frequently
2. **Reduced Exploration**: Less time wandering through unprofitable periods
3. **Pattern Recognition**: Learns what "good setups" look like
4. **Curriculum Effect**: Starts with easier wins, builds confidence

## Success Criteria

Current defaults (configurable):
- **min_pnl**: 2.0% (trades must be profitable)
- **max_duration**: 50 steps (quick winners only, <4 hours)

These can be adjusted in `record_trade()` method.

## Usage

### In Environment:
```python
# Automatic - just create environment normally
env = SimpleTradingEnv(data, expert_trades_path='expert_trades.pkl')

# Expert trades loaded automatically from disk
# Recording happens automatically during trading
```

### After Training:
```python
# Get statistics
print(env.expert_replay.get_statistics())
# {'total_trades': 47, 'avg_pnl': 3.2%, 'avg_duration': 28 steps, ...}

# Manual save
env.expert_replay.save()
```

### Standalone Usage:
```python
from environments.expert_trade_replay import ExpertTradeReplay

replay = ExpertTradeReplay('my_trades.pkl')
print(f"Loaded {len(replay)} expert trades")
print(replay.get_statistics())

# Add new trade
replay.record_trade(
    entry_step=1000,
    exit_step=1025,
    trade_object=trade_dict,
    actions=action_list,
    pnl_percent=3.5,
    hold_duration=25
)
replay.save()
```

## Files

- `src/environments/expert_trade_replay.py` - Main replay class
- `src/environments/simple_trading_env.py` - Integration with environment
- `expert_trades.pkl` - Persisted expert trades (created after first training)
- `RL_BACK.ipynb` - Training notebook with save logic

## Future Enhancements

Possible improvements:
- **Adaptive injection rate**: Increase as model improves
- **Difficulty progression**: Start with best trades, gradually add harder ones
- **Action hints**: Give model hints about expert actions (imitation learning)
- **Validation**: Track if model successfully replicates expert trades
- **Multi-criteria**: Record trades meeting different success patterns
