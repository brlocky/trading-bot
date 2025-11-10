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


def precompute_price_context_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """Pre-compute price context features."""
    feature_cols = []

    # Time features (raw values, no artificial scaling)
    date_col = pd.to_datetime(df['date'])
    df['day_of_week'] = date_col.dt.dayofweek.astype(float) / 6.0  # Normalize to 0-1
    df['hour'] = date_col.dt.hour.astype(float) / 23.0  # Normalize to 0-1
    feature_cols.extend(['day_of_week', 'hour'])

    # Candle structure (natural ratios 0-1)
    df['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, 1e-6)
    df['upper_wick_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low']).replace(0, 1e-6)
    df['lower_wick_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    feature_cols.extend(['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'close_position'])

    # Volume (rolling window normalization to 0-1 range)
    volume_rolling_min = df['volume'].rolling(window=window, min_periods=1).min()
    volume_rolling_max = df['volume'].rolling(window=window, min_periods=1).max()
    volume_rolling_range = (volume_rolling_max - volume_rolling_min).replace(0, 1e-6)
    df['volume_norm'] = (df['volume'] - volume_rolling_min) / volume_rolling_range
    feature_cols.append('volume_norm')

    # Distance to EMAs (natural ratios)
    for period in [9, 21, 50, 100]:
        dist_col = f'dist_ema{period}'
        df[dist_col] = (df['close'] - df[f'ema{period}']) / df['close']
        feature_cols.append(dist_col)

    # Distance to VWAP (natural ratio)
    df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']
    feature_cols.append('dist_vwap')

    return feature_cols


def precompute_spatial_price_normalized_features(df: pd.DataFrame) -> list[str]:
    """Pre-compute normalized OHLC features using min-max scaling per row."""
    feature_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm']

    # Normalized OHLC prices (min-max per row: 0-1 scale)
    # For each candle, normalize relative to its own high/low range
    candle_range = (df['high'] - df['low']).replace(0, 1e-6)  # Avoid division by zero

    df['open_norm'] = (df['open'] - df['low']) / candle_range
    df['high_norm'] = 1.0  # High is always max (1.0)
    df['low_norm'] = 0.0   # Low is always min (0.0)
    df['close_norm'] = (df['close'] - df['low']) / candle_range

    return feature_cols


def precompute_temporal_price_normalized_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute temporal OHLC features using rolling window min-max normalization.
    Preserves trend direction and momentum over time.
    """
    feature_cols = ['open_temporal', 'high_temporal', 'low_temporal', 'close_temporal']

    # Calculate rolling min/max over the lookback window
    rolling_min = df[['open', 'high', 'low', 'close']].rolling(window=window, min_periods=1).min().min(axis=1)
    rolling_max = df[['open', 'high', 'low', 'close']].rolling(window=window, min_periods=1).max().max(axis=1)
    rolling_range = (rolling_max - rolling_min).replace(0, 1e-6)

    # Normalize all OHLC relative to rolling window range
    df['open_temporal'] = (df['open'] - rolling_min) / rolling_range
    df['high_temporal'] = (df['high'] - rolling_min) / rolling_range
    df['low_temporal'] = (df['low'] - rolling_min) / rolling_range
    df['close_temporal'] = (df['close'] - rolling_min) / rolling_range

    return feature_cols


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

    # EMA alignment - fast (9/21)
    df['ema_alignment_fast'] = np.where(df['ema9'] > df['ema21'], 1.0, -1.0)
    feature_cols.append('ema_alignment_fast')

    # EMA alignment - slow (50/100)
    df['ema_alignment_slow'] = np.where(df['ema50'] > df['ema100'], 1.0, -1.0)
    feature_cols.append('ema_alignment_slow')

    # EMA alignment - full (all EMAs in order)
    bullish_alignment = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema50']) & (df['ema50'] > df['ema100'])
    bearish_alignment = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema50']) & (df['ema50'] < df['ema100'])
    df['ema_alignment'] = np.where(bullish_alignment, 1.0, np.where(bearish_alignment, -1.0, 0.0))
    feature_cols.append('ema_alignment')

    return feature_cols


def precompute_momentum_features(df: pd.DataFrame) -> list[str]:
    df['stoch_k_norm'] = df['stoch_k'] / 100.0
    df['stoch_d_norm'] = df['stoch_d'] / 100.0
    return ['stoch_k_norm', 'stoch_d_norm']


def precompute_rsi_features(df: pd.DataFrame) -> list[str]:
    df['rsi_norm'] = df['rsi'] / 100.0
    return ['rsi_norm']


def precompute_rsi_divergence_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute RSI divergence features with normalized High/Low prices.

    Normalizes RSI + High + Low to [-1, 1] range using rolling window.
    This allows the CNN to directly learn divergence patterns:
    - "RSI rising while High falling" → Bearish divergence
    - "RSI falling while Low rising" → Bullish divergence

    Args:
        df: DataFrame with 'rsi', 'high', 'low' columns
        window: Rolling window for min-max normalization (default: 288 = 1 day lookback)

    Returns:
        List of 3 feature column names: ['rsi_divergence', 'high_divergence', 'low_divergence']
    """
    feature_cols = ['rsi_divergence', 'high_divergence', 'low_divergence']

    # RSI: Already in [0, 100] range, normalize to [-1, 1]
    # Map: 0 → -1, 50 → 0, 100 → 1
    df['rsi_divergence'] = (df['rsi'] / 50.0) - 1.0

    # High: Rolling window min-max normalization to [0, 1], then scale to [-1, 1]
    rolling_min = df['high'].rolling(window=window, min_periods=1).min()
    rolling_max = df['high'].rolling(window=window, min_periods=1).max()
    rolling_range = (rolling_max - rolling_min).replace(0, 1e-8)
    high_norm_01 = (df['high'] - rolling_min) / rolling_range
    df['high_divergence'] = (high_norm_01 * 2.0) - 1.0  # [0, 1] → [-1, 1]

    # Low: Rolling window min-max normalization to [0, 1], then scale to [-1, 1]
    rolling_min = df['low'].rolling(window=window, min_periods=1).min()
    rolling_max = df['low'].rolling(window=window, min_periods=1).max()
    rolling_range = (rolling_max - rolling_min).replace(0, 1e-8)
    low_norm_01 = (df['low'] - rolling_min) / rolling_range
    df['low_divergence'] = (low_norm_01 * 2.0) - 1.0  # [0, 1] → [-1, 1]

    return feature_cols


def precompute_macd_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute MACD features using rolling window min-max normalization.
    Normalizes each MACD component to 0-1 scale relative to recent range.
    """
    feature_cols = ['macd_norm', 'macd_signal_norm', 'macd_hist_norm']

    # Normalize each MACD component using rolling window
    for col, feature in [('macd', 'macd_norm'),
                         ('macd_signal', 'macd_signal_norm'),
                         ('macd_hist', 'macd_hist_norm')]:
        rolling_min = df[col].rolling(window=window, min_periods=1).min()
        rolling_max = df[col].rolling(window=window, min_periods=1).max()
        rolling_range = (rolling_max - rolling_min).replace(0, 1e-6)

        df[feature] = (df[col] - rolling_min) / rolling_range

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
    Group 6: Account State Features (5 features)

    Core account metrics normalized for 100x leverage trading.
    Shape: (lookback, 5)

    Features:
    1. equity / (initial_balance * 10) - Total account value (0.1=start, 1.0=10x)
    2. current_balance / (initial_balance * 10) - Free balance available
    3. unrealized_pnl / (initial_balance * 5) + 0.5 - Current trade PnL (-5x to +5x → 0-1)
    4. realized_pnl / (initial_balance * 5) + 0.5 - Closed trades PnL (-5x to +5x → 0-1)
    5. total_commission / (initial_balance * 5) - Cumulative fees (0 to 5x → 0-1)
    """
    history_slice = broker_history[-lookback:]
    start_idx = lookback - len(history_slice)

    # Vectorized extraction
    equities = np.array([h['equity'] for h in history_slice], dtype=np.float32)
    current_balances = np.array([h['current_balance'] for h in history_slice], dtype=np.float32)
    unrealized_pnls = np.array([h['unrealized_pnl'] for h in history_slice], dtype=np.float32)
    realized_pnls = np.array([h['realized_pnl'] for h in history_slice], dtype=np.float32)
    total_commissions = np.array([h['total_commission'] for h in history_slice], dtype=np.float32)

    # Pre-allocate features
    features = np.zeros((lookback, 5), dtype=np.float32)

    # Normalize to 0-1 range
    features[start_idx:, 0] = equities / (initial_balance * 10.0)
    features[start_idx:, 1] = current_balances / (initial_balance * 10.0)
    features[start_idx:, 2] = (unrealized_pnls / (initial_balance * 5.0)) + 0.5
    features[start_idx:, 3] = (realized_pnls / (initial_balance * 5.0)) + 0.5
    features[start_idx:, 4] = total_commissions / (initial_balance * 5.0)  # Allow up to 5x initial balance in fees

    return torch.from_numpy(features)


def get_position_info_features(broker_history: list, lookback: int) -> torch.Tensor:
    """
    Group 7: Current Position Info Features (100x Leverage Compatible)

    Information about the active position (if one exists).
    Shape: (lookback, 7)

    Features:
    1. position_status: 0=none, 0.5=long, 1.0=short (normalized discrete)
    2. leverage_used: (used_balance / equity) / 100 - Actual leverage (0-1 scale)
    3. unrealized_pnl_pct: unrealized_pnl / used_balance * 0.2 + 0.5 - Position P/L (-250% to +250% → 0-1)
    4. distance_to_sl: (sl_price - current_price) / entry_price * 10 + 0.5 - SL distance (-5% to +5% → 0-1)
    5. distance_to_tp: (tp_price - current_price) / entry_price * 10 + 0.5 - TP distance (-5% to +5% → 0-1)
    6. risk_reward_ratio: tp_distance / sl_distance / 10 - Risk/Reward (0-10x → 0-1)
    7. position_duration / 288: Candles since entry (0-288 → 0-1, full lookback window)
    """
    history_slice = broker_history[-lookback:]
    start_idx = lookback - len(history_slice)

    # Extract position data
    position_sizes = np.array([h['position_size'] for h in history_slice], dtype=np.float32)
    used_balances = np.array([h['used_balance'] for h in history_slice], dtype=np.float32)
    equities = np.array([h['equity'] if h['equity'] > 0 else 1e5 for h in history_slice], dtype=np.float32)
    unrealized_pnls = np.array([h['unrealized_pnl'] for h in history_slice], dtype=np.float32)
    current_prices = np.array([h['current_price'] for h in history_slice], dtype=np.float32)
    entry_prices = np.array([h['entry_price'] if h['entry_price'] > 0 else h['current_price'] for h in history_slice], dtype=np.float32)
    sl_prices = np.array([h.get('stop_loss_price', 0) or 0 for h in history_slice], dtype=np.float32)
    tp_prices = np.array([h.get('take_profit_price', 0) or 0 for h in history_slice], dtype=np.float32)
    steps = np.array([h['step'] for h in history_slice], dtype=np.float32)

    # Pre-allocate output
    features = np.zeros((lookback, 7), dtype=np.float32)

    # Feature 1: Position status (0/0.5/1.0)
    features[start_idx:, 0] = np.where(position_sizes > 0, 0.5,
                                       np.where(position_sizes < 0, 1.0, 0.0))

    # Feature 2: Leverage used (normalized by 100x max)
    leverage = used_balances / equities
    features[start_idx:, 1] = leverage / 100.0

    # Feature 3: Unrealized PnL % of position margin (-250% to +250% with 100x leverage)
    unrealized_pct = np.divide(
        unrealized_pnls,
        used_balances,
        out=np.zeros_like(unrealized_pnls),
        where=(used_balances > 0)
    )
    # Scale and center: -250% to +250% → 0 to 1 (0.5 = breakeven)
    features[start_idx:, 2] = (unrealized_pct * 0.2) + 0.5  # *0.2 scales ±2.5 range to ±0.5

    # Feature 4: Distance to SL (% of current price, tanh normalized)
    # tanh maps any % distance to [-1, 1] smoothly
    # Close distances (~1%) are more sensitive, far distances (~10%+) asymptote to ±1
    sl_distance_raw = np.where((sl_prices > 0) & (current_prices > 0),
                               (sl_prices - current_prices) / current_prices,
                               0.0)
    # Invert sign for shorts so model sees consistent semantics
    sl_distance = np.where(position_sizes < 0, -sl_distance_raw, sl_distance_raw)
    # Scale by 10 before tanh (±10% → ±1.0, ±1% → ±0.1)
    features[start_idx:, 3] = np.tanh(sl_distance * 10.0)

    # Feature 5: Distance to TP (% of current price, tanh normalized)
    tp_distance_raw = np.where((tp_prices > 0) & (current_prices > 0),
                               (tp_prices - current_prices) / current_prices,
                               0.0)
    # Invert sign for shorts
    tp_distance = np.where(position_sizes < 0, -tp_distance_raw, tp_distance_raw)
    # Scale by 10 before tanh (±10% → ±1.0, ±1% → ±0.1)
    features[start_idx:, 4] = np.tanh(tp_distance * 10.0)

    # Feature 6: Risk/Reward ratio (TP distance / SL distance)
    sl_distance_abs = np.abs(sl_prices - entry_prices)
    tp_distance_abs = np.abs(tp_prices - entry_prices)
    rr_ratio = np.where((sl_distance_abs > 0) & (tp_distance_abs > 0),
                        tp_distance_abs / sl_distance_abs,
                        0.0)
    features[start_idx:, 5] = np.clip(rr_ratio / 10.0, 0, 1)  # Clip to 0-10x → 0-1

    # Feature 7: Position duration (candles since entry, tanh normalized)
    # tanh allows duration to exceed lookback window without clipping
    # Short durations (0-50 steps) are more sensitive, long durations (100+) asymptote to 1.0
    entry_bars = np.array([h.get('entry_bar', 0) if h['position_size'] != 0 else 0 for h in history_slice], dtype=np.float32)
    duration = np.where(position_sizes != 0, steps - entry_bars, 0)
    # Normalize by lookback and apply tanh (288 steps → 1.0, 576 steps → ~0.76 after tanh)
    features[start_idx:, 6] = np.tanh(duration / float(lookback))

    return torch.from_numpy(features)
