# Modular Reward Components - Usage Example

## ✅ What Was Created

1. **`src/environments/reward_components.py`** - Core modular system
2. **`test_reward_components.py`** - Standalone test
3. **This guide** - Integration instructions

---

## Simple Structure

```python
# src/environments/reward_components.py

class RewardComponent:
    def calculate(action, current_state, previous_state, current_step, data) -> float:
        """Return reward in [-1, 1]"""
        pass

class ModularRewardFunction:
    def add_component(name, component, weight)
    def calculate(...) -> (total_reward, breakdown)
```

## Integration with SimpleTradingEnv

### Step 1: Update calculate_reward() method

```python
# In simple_trading_env.py

from environments.reward_components import ModularRewardFunction, VPEntryQualityComponent

class SimpleTradingEnv:
    def __init__(self, ...):
        # ... existing code ...
        
        # Setup modular reward function
        self.reward_fn = ModularRewardFunction()
        
        # Add components with weights
        self.reward_fn.add_component('vp_entry', VPEntryQualityComponent(), weight=0.2)
        # Add more components here later
    
    def calculate_reward(self, action, current_state, previous_state):
        """Calculate reward using modular components."""
        
        # Calculate total reward + breakdown
        total_reward, breakdown = self.reward_fn.calculate(
            action=action,
            current_state=current_state,
            previous_state=previous_state,
            current_step=self.current_step,
            data=self.data
        )
        
        # Optional: Log breakdown for analysis
        if self.enable_trade_logging:
            self.trade_logger.log_reward_breakdown(self.current_step, breakdown)
        
        return total_reward
```

### Step 2: No Column Adjustments Needed!

VP levels are calculated **on-the-fly** from the visible range (same as the environment does).
No need to worry about column names - it uses the OHLC data directly.

### Step 2: Test with Simple Config

```python
# Test in RL_BACK.ipynb or a notebook

# Create env with modular rewards
env = SimpleTradingEnv(
    train_data, 
    lookback_window=288,
    enable_trade_logging=True  # Enable to see breakdown
)

# Test a few steps
obs, info = env.reset()
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Check breakdown
    breakdown = env.reward_fn.get_breakdown()
    if 'vp_entry' in breakdown:
        print(f"Step {i}: VP Entry = {breakdown['vp_entry']['weighted']:.3f}")
```

## Adding More Components

### Example: Hold Duration Component

```python
class HoldDurationComponent(RewardComponent):
    """Penalize overholding positions."""
    
    def calculate(self, action, current_state, previous_state, current_step, data):
        trade_step = current_state.get('trade_step', 0)
        
        if trade_step > 100:
            # Quadratic penalty after 100 steps
            excess = trade_step - 100
            penalty = -0.0001 * (excess ** 2)
            return np.clip(penalty, -1.0, 0.0)
        
        return 0.0
```

### Add to Environment:

```python
self.reward_fn.add_component('hold_duration', HoldDurationComponent(), weight=0.1)
```

## Current Weight Recommendation

Start with your existing logic + VP quality:

```python
# In __init__:
self.reward_fn = ModularRewardFunction()

# Keep existing PnL logic (80%)
# self.reward_fn.add_component('pnl', PnLComponent(), weight=0.8)

# Add VP entry quality (20%)
self.reward_fn.add_component('vp_entry', VPEntryQualityComponent(), weight=0.2)
```

## Debugging

Print breakdown to see what's happening:

```python
reward, breakdown = self.reward_fn.calculate(...)

print(f"Total: {reward:.3f}")
for name, info in breakdown.items():
    print(f"  {name}: {info['weighted']:.3f} (raw={info['raw']:.3f}, weight={info['weight']})")
    if info['debug']:
        print(f"    Debug: {info['debug']}")
```

## Next Steps

1. **Integrate** the modular reward function into `simple_trading_env.py`
2. **Test** with a few episodes to see the breakdown
3. **Add more components** one at a time (hold duration, exit quality, etc.)
4. **Tune weights** based on TensorBoard metrics
