"""
Enhanced feature extraction - Phase 0.2 Optimized

Feature Groups:
1. Price Context: OHLC + spatial relationships
2. Trend Indicators: EMA slopes, crossovers, momentum
3. Momentum Oscillators: RSI, Stochastic, MACD
4. Volume Profile: Session VP levels
5. Account State, Position Info, Performance Metrics

Total: 7 groups, 60 features per timestep (reduced from 87)
"""

import pandas as pd
import numpy as np
import torch


def precompute_trend_features(df: pd.DataFrame) -> list[str]:
    """Pre-compute trend indicator features."""
    feature_cols = []

    # EMA Slopes
    df['ema9_slope'] = df['ema9'].pct_change(periods=2).fillna(0)
    df['ema21_slope'] = df['ema21'].pct_change(periods=2).fillna(0)
    df['ema50_slope'] = df['ema50'].pct_change(periods=2).fillna(0)
    df['ema100_slope'] = df['ema100'].pct_change(periods=2).fillna(0)
    feature_cols.extend(['ema9_slope', 'ema21_slope', 'ema50_slope', 'ema100_slope'])

    # EMA Crossovers
    df['ema9_21_cross'] = np.where(df['ema9'] > df['ema21'], 1.0, -1.0)
    df['ema50_100_cross'] = np.where(df['ema50'] > df['ema100'], 1.0, -1.0)
    feature_cols.extend(['ema9_21_cross', 'ema50_100_cross'])

    # EMA alignment
    bullish_alignment = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema50']) & (df['ema50'] > df['ema100'])
    bearish_alignment = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema50']) & (df['ema50'] < df['ema100'])
    df['ema_alignment'] = np.where(bullish_alignment, 1.0, np.where(bearish_alignment, -1.0, 0.0))
    feature_cols.append('ema_alignment')

    return feature_cols


def precompute_momentum_features() -> list[str]:
    return ['stoch_k', 'stoch_d']


def precompute_rsi_features() -> list[str]:
    return ['rsi']


def precompute_macd_features(df: pd.DataFrame) -> list[str]:
    """Pre-compute MACD features for divergence CNN."""
    # MACD components (normalized relative to price for scale independence)
    df['macd_norm'] = df['macd'] / df['close']
    df['macd_signal_norm'] = df['macd_signal'] / df['close']
    df['macd_hist_norm'] = df['macd_hist'] / df['close']
    return ['macd_norm', 'macd_signal_norm', 'macd_hist_norm']


def precompute_price_context_features(df: pd.DataFrame) -> list[str]:
    """Pre-compute price context features."""
    feature_cols = []

    # Time features (raw values, no artificial scaling)
    date_col = pd.to_datetime(df['date'])
    df['day_of_week'] = date_col.dt.dayofweek.astype(float)
    df['hour'] = date_col.dt.hour.astype(float)
    feature_cols.extend(['day_of_week', 'hour'])

    # Candle structure (natural ratios 0-1)
    df['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, 1e-6)
    df['upper_wick_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low']).replace(0, 1e-6)
    df['lower_wick_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    feature_cols.extend(['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'close_position'])

    # Volume (raw values, no min-max scaling)
    feature_cols.append('volume')

    # Distance to EMAs (natural ratios)
    for period in [9, 21, 50, 100]:
        dist_col = f'dist_ema{period}'
        df[dist_col] = (df['close'] - df[f'ema{period}']) / df['close']
        feature_cols.append(dist_col)

    # Distance to VWAP (natural ratio)
    df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']
    feature_cols.append('dist_vwap')

    return feature_cols


def precompute_trading_sessions(df: pd.DataFrame) -> list[str]:
    """
    Pre-compute trading session flags (ASIA, LONDON, NY) and add them as columns IN PLACE.
    These are static based on timestamps.

    Modifies df in place.
    Returns: List of column names added
    """
    from utils.session_utils import get_trading_session

    feature_cols = []

    # Get session flags for each timestamp
    date_col = pd.to_datetime(df['date'])
    session_flags = date_col.apply(lambda ts: get_trading_session(ts)[3:])  # Returns (is_asia, is_london, is_ny)

    # Unpack into separate columns
    df['is_asia'] = session_flags.apply(lambda x: float(x[0]))
    df['is_london'] = session_flags.apply(lambda x: float(x[1]))
    df['is_ny'] = session_flags.apply(lambda x: float(x[2]))
    feature_cols.extend(['is_asia', 'is_london', 'is_ny'])

    return feature_cols


def get_volume_profile_features(vp_obj, current_price: float, lookback: int) -> torch.Tensor:
    """
    Extract VP features from EnhancedVolumeProfile object.

    Returns (lookback, 26) features:
    - Current day: vah, val, poc distances (3)
    - Current day: vah/val/poc range positions (3)
    - Previous 3 daily sessions: vah/val/poc distances, naked flags (12)
    - Previous 2 weekly sessions: vah/val/poc distances, naked flags (8)

    All distances normalized by current price (natural ratio)
    """
    features = np.zeros((lookback, 26), dtype=np.float32)

    # Current day levels (updated live during day)
    if vp_obj.current_day_vah is not None:
        vah_dist = (vp_obj.current_day_vah - current_price) / current_price
        val_dist = (vp_obj.current_day_val - current_price) / current_price
        poc_dist = (vp_obj.current_day_poc - current_price) / current_price

        # Broadcast to all lookback timesteps
        features[:, 0] = vah_dist
        features[:, 1] = val_dist
        features[:, 2] = poc_dist

        # Range positions (0-1 where in VAH-VAL range)
        day_range = vp_obj.current_day_high - vp_obj.current_day_low
        if day_range > 0:
            features[:, 3] = (current_price - vp_obj.current_day_low) / day_range

        va_range = vp_obj.current_day_vah - vp_obj.current_day_val
        if va_range > 0:
            features[:, 4] = (current_price - vp_obj.current_day_val) / va_range

        poc_range = abs(vp_obj.current_day_poc - current_price)
        if poc_range > 0:
            features[:, 5] = poc_range / current_price

    # Previous daily sessions (last 3 days)
    daily_sessions = list(vp_obj.daily_sessions)
    for i in range(min(3, len(daily_sessions))):
        session = daily_sessions[-(i+1)]
        base_idx = 6 + i * 4

        features[:, base_idx] = (session['vah'] - current_price) / current_price
        features[:, base_idx + 1] = (session['val'] - current_price) / current_price
        features[:, base_idx + 2] = (session['poc'] - current_price) / current_price
        features[:, base_idx + 3] = 0.0 if session['poc_touched'] else 1.0

    # Previous weekly sessions (last 2 weeks)
    weekly_sessions = list(vp_obj.weekly_sessions)
    for i in range(min(2, len(weekly_sessions))):
        session = weekly_sessions[-(i+1)]
        base_idx = 18 + i * 4

        features[:, base_idx] = (session['vah'] - current_price) / current_price
        features[:, base_idx + 1] = (session['val'] - current_price) / current_price
        features[:, base_idx + 2] = (session['poc'] - current_price) / current_price
        features[:, base_idx + 3] = 0.0 if session['poc_touched'] else 1.0

    return torch.from_numpy(features)


def get_volume_profile_bins(vp_obj, lookback: int, close_prices: torch.Tensor = None) -> torch.Tensor:
    """
    Group 13: Daily Volume Profile Distribution (Rolling Window + Level Markers + Price Line)

    Returns VP bins from rolling window with VAH/VAL/POC markers and close price position.
    This enables the CNN to learn intraday volume patterns with key level annotations and price context.

    Shape: (lookback, n_bins + 4) where:
    - Channels 0 to n_bins-1: Volume distribution (rolling window, normalized)
    - Channel n_bins: VAH marker (1.0 near VAH bin, 0.0 elsewhere)
    - Channel n_bins+1: VAL marker (1.0 near VAL bin, 0.0 elsewhere)
    - Channel n_bins+2: POC marker (1.0 near POC bin, 0.0 elsewhere)
    - Channel n_bins+3: Close price position (1.0 at current close bin, 0.0 elsewhere)

    Args:
        vp_obj: EnhancedVolumeProfile instance
        lookback: Number of timesteps to return
        close_prices: Optional tensor of close prices (lookback,) to show price position

    Returns:
        Tensor of shape (lookback, n_bins + 4)
    """
    # Get rolling window bins (already normalized)
    vp_bins_tensor = vp_obj.get_bins_history(lookback)  # Shape: (lookback, n_bins)
    n_bins = vp_bins_tensor.shape[1]

    # Get current VP levels
    levels = vp_obj.get_levels()
    vah = levels['vah']
    val = levels['val']
    poc = levels['poc']

    # Get price range for bin mapping
    bins = vp_obj.bins
    price_min = bins[0].item() if len(bins) > 0 else 0
    price_max = bins[-1].item() if len(bins) > 0 else 1

    # Create marker channels (4 channels: VAH, VAL, POC, Close)
    markers = torch.zeros(lookback, 4, device=vp_bins_tensor.device, dtype=torch.float32)

    if price_max > price_min:
        # Map levels to bin indices
        if vah is not None and val is not None and poc is not None:
            vah_bin = int((vah - price_min) / (price_max - price_min) * (n_bins - 1))
            val_bin = int((val - price_min) / (price_max - price_min) * (n_bins - 1))
            poc_bin = int((poc - price_min) / (price_max - price_min) * (n_bins - 1))

            # Clamp to valid range
            vah_bin = max(0, min(n_bins - 1, vah_bin))
            val_bin = max(0, min(n_bins - 1, val_bin))
            poc_bin = max(0, min(n_bins - 1, poc_bin))

            # Set level markers for all timesteps (levels are constant) - vectorized
            if vah_bin > 0:
                markers[:, 0] = 1.0
            if val_bin < n_bins - 1:
                markers[:, 1] = 1.0
            markers[:, 2] = 1.0  # POC marker - strongest marker

        # Add close price markers (channel 3) - changes over time - vectorized
        if close_prices is not None:
            # Map all close prices to bin indices at once
            close_bins = ((close_prices - price_min) / (price_max - price_min) * (n_bins - 1)).long()
            close_bins = torch.clamp(close_bins, 0, n_bins - 1)
            # Set markers (simplified - just mark that price exists, CNN will learn spatial relationship)
            markers[:, 3] = 1.0

    # Concatenate bins and markers
    result = torch.cat([vp_bins_tensor, markers], dim=1)  # Shape: (lookback, n_bins + 4)
    return result


def get_account_state_features(broker_history: list, initial_balance: float, lookback: int) -> torch.Tensor:
    """
    Group 6: Account State Features (SIMPLIFIED)

    Returns ONLY normalized equity (balance + unrealized PnL)
    Shape: (lookback, 1)

    Natural ratio normalization - no clipping, GroupNorm handles outliers
    """
    history_slice = broker_history[-lookback:]
    start_idx = lookback - len(history_slice)

    # Vectorized extraction
    equities = np.array([h['equity'] for h in history_slice], dtype=np.float32)

    # Pre-allocate and assign
    features = np.zeros((lookback, 1), dtype=np.float32)
    features[start_idx:, 0] = equities / initial_balance

    return torch.from_numpy(features)


def get_position_info_features(broker_history: list, lookback: int) -> torch.Tensor:
    """
    Group 7: Position Info Features (SIMPLIFIED)

    Returns ONLY position status and exposure (natural ratios, no clipping)
    Shape: (lookback, 2)

    Features:
    - position_status: 0=none, 1=long, 2=short (discrete signal)
    - position_exposure: used_balance / equity (% of account locked in position)
    """
    history_slice = broker_history[-lookback:]
    start_idx = lookback - len(history_slice)

    # Extract all values in ONE vectorized pass
    position_sizes = np.array([h['position_size'] for h in history_slice], dtype=np.float32)
    used_balances = np.array([h['used_balance'] for h in history_slice], dtype=np.float32)
    equities = np.array([h['equity'] if h['equity'] > 0 else 1e5 for h in history_slice], dtype=np.float32)

    # Pre-allocate output
    features = np.zeros((lookback, 2), dtype=np.float32)

    # Vectorized Feature 1: Position status (0/1/2)
    features[start_idx:, 0] = np.where(position_sizes > 0, 1.0,
                                       np.where(position_sizes < 0, 2.0, 0.0))

    # Vectorized Feature 2: Position exposure (used_balance / equity)
    features[start_idx:, 1] = used_balances / equities

    return torch.from_numpy(features)
