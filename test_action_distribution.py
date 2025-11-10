"""
Debug: Check model's action distribution and probabilities
"""
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.simple_trading_env import SimpleTradingEnv
import numpy as np
import sys
sys.path.append('src')

# Load data
df = pd.read_pickle('data/binance-BTCUSDT-5m.pkl')
test_data = df.iloc[5_536:5_536 + 500].reset_index(drop=True)

# Create environment
test_env = SimpleTradingEnv(test_data, device="cuda", lookback_window=288)
test_env = Monitor(test_env)
test_env = DummyVecEnv([lambda: test_env])

# Load model
model = PPO.load("trading_bot", env=test_env, device="cuda")

# Track actions and probabilities
obs = test_env.reset()
action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
action_probs_sum = {0: [], 1: [], 2: [], 3: []}

print("\n=== SAMPLING MODEL ACTIONS ===\n")
for step in range(100):
    # Get action probabilities
    with torch.no_grad():
        obs_tensor = {
            key: torch.tensor(val, dtype=torch.float32, device=model.device)
            for key, val in obs.items()
        }

        # Get action distribution
        distribution = model.policy.get_distribution(obs_tensor)

        # For MultiDiscrete, distribution.distribution is a list of Categorical distributions
        direction_dist = distribution.distribution[0]  # First action (direction)
        direction_probs = direction_dist.probs[0].cpu().numpy()  # Get probabilities

        # Record probabilities
        for i in range(4):
            action_probs_sum[i].append(direction_probs[i])

    # Get action
    action, _states = model.predict(obs, deterministic=True)
    direction_action = int(action[0][0])
    action_counts[direction_action] += 1

    # Step
    obs, reward, done, info = test_env.step(action)
    if done[0]:
        break

# Print results
print(f"Actions taken (100 steps):")
print(f"  0 (HOLD):  {action_counts[0]:3d} ({action_counts[0]/100*100:5.1f}%)")
print(f"  1 (LONG):  {action_counts[1]:3d} ({action_counts[1]/100*100:5.1f}%)")
print(f"  2 (SHORT): {action_counts[2]:3d} ({action_counts[2]/100*100:5.1f}%)")
print(f"  3 (CLOSE): {action_counts[3]:3d} ({action_counts[3]/100*100:5.1f}%)")

print(f"\nAverage action probabilities:")
for i in range(4):
    avg_prob = np.mean(action_probs_sum[i]) if action_probs_sum[i] else 0
    action_names = ['HOLD', 'LONG', 'SHORT', 'CLOSE']
    print(f"  {action_names[i]:6s}: {avg_prob:.3f} ({avg_prob*100:.1f}%)")

print(f"\nDiagnosis:")
long_prob = np.mean(action_probs_sum[1]) if action_probs_sum[1] else 0
short_prob = np.mean(action_probs_sum[2]) if action_probs_sum[2] else 0
if short_prob > long_prob * 2:
    print(f"  WARNING: Model strongly prefers SHORT (ratio: {short_prob/long_prob:.2f}:1)")
    print(f"  Likely cause: Reward function asymmetry or feature bias")
elif long_prob > short_prob * 2:
    print(f"  WARNING: Model strongly prefers LONG (ratio: {long_prob/short_prob:.2f}:1)")
else:
    print(f"  OK: Model has balanced directional preferences")
