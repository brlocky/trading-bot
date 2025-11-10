# Divergence Utils - New API Examples

This notebook demonstrates the new precomputed divergence API.

## Quick Start

```python
import pandas as pd
from utils import add_indicators, add_zigzag, add_divergence

# Load your data
df = pd.read_json('data/BTCUSDT-15m.json')

# Step 1: Add indicators
add_indicators(df)

# Step 2: Add zigzag columns with different thresholds
add_zigzag(df, threshold=0.5, column_name='zigzag_0_5')
add_zigzag(df, threshold=1.0, column_name='zigzag_1_0')
add_zigzag(df, threshold=2.0, column_name='zigzag_2_0')

# Step 3: Add divergence columns for RSI
add_divergence(
    df,
    indicator_column='rsi',
    zigzag_columns=['zigzag_0_5', 'zigzag_1_0', 'zigzag_2_0'],
    price_column='close'
)

# This creates columns:
# - rsi_div_zigzag_0_5
# - rsi_div_zigzag_1_0
# - rsi_div_zigzag_2_0

# Step 4: Add divergence columns for MACD
add_divergence(
    df,
    indicator_column='macd',
    zigzag_columns=['zigzag_0_5', 'zigzag_1_0', 'zigzag_2_0'],
    price_column='close'
)

# This creates columns:
# - macd_div_zigzag_0_5
# - macd_div_zigzag_1_0
# - macd_div_zigzag_2_0
```

## Divergence Values

Each divergence column contains integers indicating the divergence type:
- **0**: No divergence
- **1**: Regular Bullish (reversal signal - expect price to go up)
- **2**: Regular Bearish (reversal signal - expect price to go down)
- **3**: Hidden Bullish (continuation signal - expect uptrend to continue)
- **4**: Hidden Bearish (continuation signal - expect downtrend to continue)

## Example Usage in Features

```python
# Check for any divergence at current step
if df.loc[current_step, 'rsi_div_zigzag_1_0'] > 0:
    print(f"RSI Divergence detected!")
    
# Check for specific divergence type
if df.loc[current_step, 'rsi_div_zigzag_1_0'] == 1:
    print("Regular Bullish divergence - potential reversal up!")

# Count divergences in a window
window = df.iloc[current_step-50:current_step+1]
bullish_divs = (window['rsi_div_zigzag_1_0'] == 1).sum()
bearish_divs = (window['rsi_div_zigzag_1_0'] == 2).sum()
```

## Custom Column Prefix

```python
# Use custom prefix for divergence columns
add_divergence(
    df,
    indicator_column='rsi',
    zigzag_columns=['zigzag_1_0'],
    column_prefix='rsi_divergence'
)

# Creates: rsi_divergence_zigzag_1_0
```

## Multiple Indicators

```python
# Precompute all divergences at once
indicators = ['rsi', 'macd', 'macd_hist']
zigzag_cols = ['zigzag_0_5', 'zigzag_1_0', 'zigzag_2_0', 'zigzag_5_0']

for indicator in indicators:
    add_divergence(
        df,
        indicator_column=indicator,
        zigzag_columns=zigzag_cols
    )

# Now you have divergence columns for all indicators at all thresholds
# Total: 3 indicators × 4 thresholds = 12 divergence columns
```

## Benefits

1. **Precomputed**: Calculate once, use many times - no runtime overhead
2. **Multiple thresholds**: Test different zigzag sensitivities
3. **Clean data**: Divergences stored as simple integer columns
4. **Easy filtering**: Use pandas filtering to find divergence candles
5. **Memory efficient**: Integer columns (int8) use minimal space

## Integration with Trading Environment

```python
# In your data preprocessing:
def prepare_trading_data(df):
    # Add indicators
    add_indicators(df)
    
    # Add zigzag at multiple scales
    add_zigzag(df, threshold=0.5, column_name='zigzag_0_5')
    add_zigzag(df, threshold=1.0, column_name='zigzag_1_0')
    add_zigzag(df, threshold=2.0, column_name='zigzag_2_0')
    
    # Precompute divergences
    for indicator in ['rsi', 'macd']:
        add_divergence(
            df,
            indicator_column=indicator,
            zigzag_columns=['zigzag_0_5', 'zigzag_1_0', 'zigzag_2_0']
        )
    
    return df

# Use in environment
df = prepare_trading_data(raw_df)

# Access divergence features in your observation space
divergence_features = [
    'rsi_div_zigzag_0_5',
    'rsi_div_zigzag_1_0',
    'rsi_div_zigzag_2_0',
    'macd_div_zigzag_0_5',
    'macd_div_zigzag_1_0',
    'macd_div_zigzag_2_0',
]
```
