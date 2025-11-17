"""
Quick test of modular reward components
"""

import pandas as pd
from environments.reward_components import ModularRewardFunction, VPEntryQualityComponent

# Load data
df = pd.read_pickle('data/binance-BTCUSDT-5m.pkl')
test_data = df.iloc[1000:2000].reset_index(drop=True)

# Create reward function
reward_fn = ModularRewardFunction()
reward_fn.add_component('vp_entry', VPEntryQualityComponent(), weight=1.0)

# Simulate some actions
print("Testing VP Entry Quality Component:\n")

for step in [50, 100, 150, 200, 250]:
    # Simulate LONG entry
    action = 1
    current_state = {'position_size': 100}  # After entry
    previous_state = {'position_size': 0}   # Before entry

    reward, breakdown = reward_fn.calculate(
        action=action,
        current_state=current_state,
        previous_state=previous_state,
        current_step=step,
        data=test_data
    )

    print(f"Step {step}: LONG entry")
    print(f"  Reward: {reward:.3f}")
    if 'vp_entry' in breakdown:
        debug = breakdown['vp_entry']['debug']
        if 'close' in debug:
            print(f"  Price: {debug['close']:.2f}")
            print(f"  VAL: {debug['val']:.2f}, POC: {debug['poc']:.2f}, VAH: {debug['vah']:.2f}")
        else:
            print(f"  Reason: {debug.get('reason', 'unknown')}")
    print()
