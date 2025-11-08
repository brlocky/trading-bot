"""
Indicator Utils - Phase 0.2 Optimized
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add minimal high-signal indicators only."""
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_9'] = ta.rsi(df['close'], length=9)

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    df['volume_ma20'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    df['volatility'] = ta.stdev(df['close'], length=20) / df['close']

    # Used on the Stop Loss/Take Profit calculations
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    ema_periods = [9, 21, 50, 100]
    for p in ema_periods:
        df[f'ema{p}'] = ta.ema(df['close'], length=p)

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3)
    df['stoch_k'] = stoch['STOCHk_5_3_3']
    df['stoch_d'] = stoch['STOCHd_5_3_3']

    days = df['date'].dt.floor('D')
    cum_vol = df.groupby(days)['volume'].cumsum()
    cum_vp = (df['volume'] * df['close']).groupby(days).cumsum()
    num = cum_vp.to_numpy(dtype=float)
    den = cum_vol.to_numpy(dtype=float)
    df['vwap'] = np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)

    return df
