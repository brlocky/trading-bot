import pandas as pd

df = pd.read_pickle('data/binance-BTCUSDT-5m.pkl')

# Training data
train = df.iloc[0:32_768].copy()
train_start = train['close'].iloc[0]
train_end = train['close'].iloc[-1]
train_return = (train_end - train_start) / train_start * 100

# Test data
test = df.iloc[5_536:5_536+2000].copy()
test_start_price = test['close'].iloc[0]
test_end_price = test['close'].iloc[-1]
test_return = (test_end_price - test_start_price) / test_start_price * 100

print(f'MARKET COMPARISON:')
print(f'\nTraining Data (0 to 32,768):')
print(f'  Start: {train_start:.2f} -> End: {train_end:.2f}')
print(f'  Return: {train_return:+.2f}%')
print(f'  Up bars: {(train["close"] > train["open"]).sum()} ({(train["close"] > train["open"]).sum()/len(train)*100:.1f}%)')

print(f'\nTest Data (5,536 to 7,536):')
print(f'  Start: {test_start_price:.2f} -> End: {test_end_price:.2f}')
print(f'  Return: {test_return:+.2f}%')
print(f'  Up bars: {(test["close"] > test["open"]).sum()} ({(test["close"] > test["open"]).sum()/len(test)*100:.1f}%)')

print(f'\nAnalysis:')
if test_return < -2:
    print(f'  WARNING: Test period is BEARISH ({test_return:+.2f}%)')
    print(f'  Model may be correctly predicting downtrend, not biased!')
elif test_return > 2:
    print(f'  WARNING: Test period is BULLISH ({test_return:+.2f}%)')
    print(f'  If model shorts here, it is biased')
else:
    print(f'  OK: Test period is NEUTRAL ({test_return:+.2f}%)')
