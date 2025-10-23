
import numpy as np
import pandas as pd


def get_vp_levels(prices, volumes, bins=50, value_area=0.68) -> dict[str, float]:
    """Compute POC, VAH, VAL for a single slice of past data."""
    hist, edges = np.histogram(prices, bins=bins, weights=volumes)
    if hist.sum() == 0:
        return {'poc': np.nan, 'vah': np.nan, 'val': np.nan}

    poc_idx = np.argmax(hist)
    poc = (edges[poc_idx] + edges[poc_idx+1]) / 2

    # Value Area
    sorted_idx = np.argsort(hist)[::-1]
    cum_vol = np.cumsum(hist[sorted_idx])
    cutoff = hist.sum() * value_area
    sel = sorted_idx[cum_vol <= cutoff]
    sel_edges = np.r_[edges[min(sel, default=poc_idx)], edges[max(sel, default=poc_idx)+1]]
    val, vah = sel_edges[0], sel_edges[-1]

    return {'poc': poc, 'vah': vah, 'val': val}


def get_price_features(df):
    """
    Return price-derived features for LSTM:
    - High, Low, Open relative to Close
    - Close relative to Low
    """
    df = df.copy()

    df['high_close'] = (df['high'] - df['close']) / df['close']
    df['low_close'] = (df['close'] - df['low']) / df['close']
    df['open_close'] = (df['open'] - df['close']) / df['close']
    df['close_low'] = (df['close'] - df['low']) / df['low']

    date_col = pd.to_datetime(df['date'])

    # Normalize day_of_week (0-6) to [0, 1]
    df['day_of_week'] = date_col.dt.dayofweek / 6.0
    # Normalize hour (0-23) to [0, 1]
    df['hour'] = date_col.dt.hour / 23.0

    feature_cols = ['high_close', 'low_close', 'open_close', 'close_low', 'day_of_week', 'hour']

    if df[feature_cols].isnull().any().any():
        raise ValueError("NaN detected in price features! Check OHLC input.")

    return df[feature_cols]


def get_indicator_features(df):

    df = df.copy()

    # --- Relative features
    df['ema9_dist'] = (df['close'] - df['ema9']) / df['close']
    df['ema21_dist'] = (df['close'] - df['ema21']) / df['close']
    df['ema50_dist'] = (df['close'] - df['ema50']) / df['close']
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['close']
    df['return'] = df['close'].pct_change().fillna(0)
    df['volume_norm'] = (df['volume'] - df['volume'].rolling(3).mean()) / (df['volume'].rolling(3).std() + 1e-9)

    # --- Above/below binary features
    df['above_ema9'] = (df['close'] > df['ema9']).astype(int)
    df['above_ema21'] = (df['close'] > df['ema21']).astype(int)
    df['above_ema50'] = (df['close'] > df['ema50']).astype(int)
    df['above_vwap'] = (df['close'] > df['vwap']).astype(int)

    # --- Cross signals (1 = bullish cross, -1 = bearish cross, 0 = none)
    def cross_signal(short, long):
        return ((short > long) & (short.shift(1) <= long.shift(1))).astype(int) - \
               ((short < long) & (short.shift(1) >= long.shift(1))).astype(int)

    df['ema9_21_cross'] = cross_signal(df['ema9'], df['ema21'])
    df['ema21_50_cross'] = cross_signal(df['ema21'], df['ema50'])
    df['price_vwap_cross'] = cross_signal(df['close'], df['vwap'])

    # --- Clean up
    feature_cols = [
        'return', 'ema9_dist', 'ema21_dist', 'ema50_dist', 'vwap_dist', 'volume_norm',
        'above_ema9', 'above_ema21', 'above_ema50', 'above_vwap',
        'ema9_21_cross', 'ema21_50_cross', 'price_vwap_cross'
    ]

    # Drop rows with NaNs
    original_len = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    dropped = original_len - len(df_clean)
    print(f"Dropped {dropped} rows containing NaNs" if dropped > 0 else "No rows dropped")

    return df_clean[feature_cols]
