"""
KISS Indicator Utils
Minimal shared indicator calculation for both MLTrainer and trade generator.
"""

import pandas as pd
import numpy as np
import talib


def add_indicators(df: pd.DataFrame, add_crossovers: bool = True) -> pd.DataFrame:
    """Add TA-Lib indicators and derived features to a price DataFrame.

    Parameters:
        df: OHLCV DataFrame with columns ['open','high','low','close','volume','time']
        add_crossovers: whether to add EMA crossover binary flags
    """
    close = df['close'].astype(float).to_numpy()
    high = df['high'].astype(float).to_numpy()
    low = df['low'].astype(float).to_numpy()
    volume = df['volume'].astype(float).to_numpy()

    # === RSI ===
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # === MACD ===
    macd, macd_signal, macd_history = talib.MACD(close)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_history

    # === Bollinger Bands ===
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_middle'] = bb_middle

    # === Volume-based ===
    df['volume_ma20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / df['volume_ma20']
    df['obv'] = talib.OBV(close, volume)

    # === Volatility ===
    df['volatility'] = talib.STDDEV(close, timeperiod=20) / close
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)

    # === EMAs ===
    ema_periods = [5, 9, 13, 20, 21, 50, 200]
    for p in ema_periods:
        df[f'ema{p}'] = talib.EMA(close, timeperiod=p)

    # === EMA crossovers (optional) ===
    if add_crossovers:
        df['ema9_ema21_cross'] = (df['ema9'] > df['ema21']).astype(int)
        df['ema20_ema50_cross'] = (df['ema20'] > df['ema50']).astype(int)

    # === Stochastic ===
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    df['stochastic_k'] = stoch_k
    df['stochastic_d'] = stoch_d
    # === VWAP (reset daily) ===
    # Convert Unix timestamps (seconds) to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Now daily reset works
    days = df['time'].dt.floor('D')

    # Compute cumulative sums per day
    cum_vol = df.groupby(days)['volume'].cumsum()
    cum_vp = (df['volume'] * df['close']).groupby(days).cumsum()

    # Assign VWAP (use numpy arrays to avoid ExtensionArray division and handle zero denominators)
    num = cum_vp.to_numpy(dtype=float)
    den = cum_vol.to_numpy(dtype=float)
    # avoid division by zero; result will be NaN where den == 0
    df['vwap'] = np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)

    return df
