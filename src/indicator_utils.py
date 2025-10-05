"""
KISS Indicator Utils
Minimal shared indicator calculation for both MLTrainer and trade generator.
"""

import pandas as pd
import talib


def add_progressive_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add TA-Lib indicators and derived features to a price DataFrame."""
    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    volume = df['volume'].astype(float).values

    df['rsi'] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, _ = talib.MACD(close)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    df['bb_position'] = df['bb_position'].clip(0, 1)
    df['volume_ma20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / df['volume_ma20']
    df['volatility'] = talib.STDDEV(close, timeperiod=20) / close
    # Fill NaN values
    df['rsi'].fillna(50, inplace=True)
    df['bb_position'].fillna(0.5, inplace=True)
    df['volume_ratio'].fillna(1.0, inplace=True)
    df['volatility'].fillna(0.02, inplace=True)
    return df
