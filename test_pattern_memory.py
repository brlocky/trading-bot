"""
Quick test to verify pattern memory implementation
"""
import pandas as pd
from environments.simple_trading_env import SimpleTradingEnv
from environments.trading_pattern_memory import TradingPatternMemory

# Load small data sample
df = pd.read_pickle('data/binance-BTCUSDT-5m.pkl')
test_data = df.iloc[0:1000].reset_index(drop=True)

print("Testing Pattern Memory Implementation...")
print("=" * 50)

# Create environment with pattern memory enabled
env = SimpleTradingEnv(
    test_data,
    device="cpu",
    lookback_window=50,
    enable_pattern_memory=True
)

print(f"✓ Environment created with pattern memory")
print(f"✓ Pattern memory enabled: {env.enable_pattern_memory}")
print(f"✓ Initial episodes: {len(env.pattern_memory.episodes)}")

# Run a few episodes
for ep in range(3):
    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < 100:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    print(f"\nEpisode {ep + 1}: {steps} steps, final balance: ${info['current_balance']:.2f}")

# Check if episodes were recorded
print(f"\n✓ Total episodes recorded: {len(env.pattern_memory.episodes)}")

if len(env.pattern_memory.episodes) > 0:
    # Test analysis
    patterns = env.pattern_memory.get_pattern_distribution()
    print(f"\n📈 Pattern Analysis:")
    print(f"   Win Rate: {patterns['win_rate']:.1%}")
    print(f"   Total Episodes: {patterns['total_episodes']}")

    # Test export
    df_analysis = env.pattern_memory.export_to_dataframe()
    print(f"\n✓ Export successful: {len(df_analysis)} rows")

    # Test save/load
    env.pattern_memory.save('test_pattern_memory.pkl')
    print(f"✓ Saved pattern memory")

    # Test loading
    test_memory = TradingPatternMemory()
    test_memory.load('test_pattern_memory.pkl')
    print(f"✓ Loaded pattern memory: {len(test_memory.episodes)} episodes")

    print("\n✅ All tests passed!")
else:
    print("\n⚠ Warning: No episodes were recorded")
