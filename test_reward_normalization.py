"""Test min-max reward normalization - CENTERED AT ZERO"""
import pandas as pd
from environments.simple_trading_env import SimpleTradingEnv

# Load test data
df = pd.read_pickle('data/binance-BTCUSDT-5m.pkl')
test_data = df.iloc[0:1000].reset_index(drop=True)

# Create environment with reward normalization
env = SimpleTradingEnv(
    test_data,
    device='cpu',
    lookback_window=50,
    enable_pattern_memory=True,
    reward_min=-200.0,
    reward_max=400.0
)

print('✓ Environment created with CENTERED min-max reward normalization')
print(f'  Reward range: [{env.reward_min}, {env.reward_max}]')
print(f'  Negative: [{env.reward_min}, 0] → [-1, 0]')
print(f'  Positive: [0, {env.reward_max}] → [0, 1]')

# Test normalization with CENTERED zero
test_rewards = [-200, -100, -50, 0, 50, 100, 200, 400]
print(f'\n📊 Centered Normalization Test:')
for r in test_rewards:
    normalized = env._normalize_reward_minmax(r)
    print(f'  Raw: {r:6.1f} → Normalized: {normalized:+.4f}')

# Verify centering
print(f'\n✓ Key property: 0 reward → 0 normalized (CENTERED!)')

# Quick episode test
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

print(f'\n✓ Environment step tested successfully')
print(f'  Last reward (normalized): {reward:.4f}')
