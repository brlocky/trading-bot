"""
Enhanced feature extraction - Multi-Scale Architecture

Feature Groups (8 groups):
1. Micro Temporal (5): OHLC + Volume [CNN - Small kernels]
2. Micro Spatial (4): Body/wick ratios [MLP - Last timestep]
3. Meso Patterns (2): 1h, 4h returns [CNN - Medium kernels]
4. Macro Patterns (1): 24h return [CNN - Large kernels]
5. Account State (5): Bounded growth/velocity features [MLP]
6. Position Info (7): Position status, leverage, distances [MLP]
7. VP Bins (n_bins): Volume distribution histogram [CNN]
8. VP Levels (3): VAH/VAL/POC distances [MLP]

Architecture rationale:
- Groups 1,3,4,7 (Temporal features): Sequences → CNN processes patterns over time
- Groups 2,8 (Spatial features): Per-candle/level structure → MLP processes last timestep
- Groups 5,6 (Trading state): Current state only → MLPs process last timestep
- Separation: Temporal (time-series) vs Spatial (per-candle) vs State (account/position)
"""

import pandas as pd
import numpy as np
import torch


def precompute_micro_temporal_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute micro temporal features (OHLC + Volume sequences).

    Computes 5 features - TRUE TIME SERIES:
    - 4 normalized OHLC (0-1): Price movement over time
    - 1 normalized volume (0-1): Trading activity over time

    These benefit from CNN temporal processing (kernels detect patterns across time).
    Modifies df in place, returns column names.
    """
    temporal_cols = []

    # Normalized OHLC (rolling window 0-1 scale)
    rolling_min = df[['open', 'high', 'low', 'close']].rolling(window=window, min_periods=1).min().min(axis=1)
    rolling_max = df[['open', 'high', 'low', 'close']].rolling(window=window, min_periods=1).max().max(axis=1)
    rolling_range = (rolling_max - rolling_min).replace(0, 1e-6)

    df['open_norm'] = (df['open'] - rolling_min) / rolling_range
    df['high_norm'] = (df['high'] - rolling_min) / rolling_range
    df['low_norm'] = (df['low'] - rolling_min) / rolling_range
    df['close_norm'] = (df['close'] - rolling_min) / rolling_range
    temporal_cols.extend(['open_norm', 'high_norm', 'low_norm', 'close_norm'])

    # Normalized volume (0-1 range)
    volume_rolling_min = df['volume'].rolling(window=window, min_periods=1).min()
    volume_rolling_max = df['volume'].rolling(window=window, min_periods=1).max()
    volume_rolling_range = (volume_rolling_max - volume_rolling_min).replace(0, 1e-6)
    df['volume_norm'] = (df['volume'] - volume_rolling_min) / volume_rolling_range
    temporal_cols.append('volume_norm')

    return temporal_cols


def precompute_micro_spatial_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute micro spatial features (per-candle structure).

    Computes 4 features - SPATIAL (no temporal dependency):
    - Body ratio: Size of candle body
    - Upper wick ratio: Top wick size
    - Lower wick ratio: Bottom wick size
    - Close position: Where close is within range

    These describe individual candles, no time relationship.
    Best processed by MLP (last timestep) or simple aggregation.
    Modifies df in place, returns column names.
    """
    spatial_cols = []

    # Candle structure (natural ratios 0-1)
    df['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, 1e-6)
    df['upper_wick_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low']).replace(0, 1e-6)
    df['lower_wick_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
    spatial_cols.extend(['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'close_position'])

    return spatial_cols


def precompute_meso_patterns_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute meso pattern features (intraday trends, medium-frequency).

    Computes 2 features:
    - 1h return (bounded): Short-term trend (4 candles)
    - 4h return (bounded): Medium-term trend (16 candles)

    Modifies df in place, returns column names.
    """
    meso_cols = []

    # Multi-timeframe returns (bounded via tanh)
    df['returns_1h'] = np.tanh(df['close'].pct_change(periods=4).fillna(0) * 100)
    df['returns_4h'] = np.tanh(df['close'].pct_change(periods=16).fillna(0) * 100)
    meso_cols.extend(['returns_1h', 'returns_4h'])

    return meso_cols


def precompute_macro_patterns_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute macro pattern features (daily trends, low-frequency).

    Computes 1 feature:
    - 24h return (bounded): Long-term trend (96 candles)

    Modifies df in place, returns column names.
    """
    macro_cols = []

    df['returns_24h'] = np.tanh(df['close'].pct_change(periods=96).fillna(0) * 100)
    macro_cols.append('returns_24h')

    return macro_cols


def precompute_market_context_features(df: pd.DataFrame, window: int = 288) -> list[str]:
    """
    Pre-compute market context features (spatial relationships, last timestep only).

    Returns 6 features:
    - 5 distance to EMAs/VWAP (bounded via tanh): current price positioning
    - 1 volatility (0-1): current market volatility state

    These features represent WHERE price is relative to key levels, not HOW it got there.
    Only current values matter, no need for temporal processing.
    """
    feature_cols = []

    # Distance to EMAs (bounded via tanh to [-1, 1])
    # Scale by 10 so ±10% distance maps to ~±1.0
    for period in [9, 21, 50, 100]:
        dist_col = f'dist_ema{period}'
        raw_distance = (df['close'] - df[f'ema{period}']) / df['close']
        df[dist_col] = np.tanh(raw_distance * 10.0)
        feature_cols.append(dist_col)

    # Distance to VWAP (bounded via tanh)
    df['dist_vwap'] = np.tanh((df['close'] - df['vwap']) / df['close'] * 10.0)
    feature_cols.append('dist_vwap')

    # Volatility (normalized to 0-1 range)
    # Use rolling std of returns as volatility proxy
    rolling_std = df['close'].pct_change().rolling(window=20, min_periods=1).std()
    rolling_std_min = rolling_std.rolling(window=window, min_periods=1).min()
    rolling_std_max = rolling_std.rolling(window=window, min_periods=1).max()
    rolling_std_range = (rolling_std_max - rolling_std_min).replace(0, 1e-6)
    df['volatility'] = ((rolling_std - rolling_std_min) / rolling_std_range).fillna(0)
    feature_cols.append('volatility')

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
    """
    Pre-compute trend indicator features with proper normalization.

    All slopes bounded via tanh, crossovers as discrete signals.
    These features show temporal momentum patterns → should use CNN encoder.
    """
    feature_cols = []

    # EMA Slopes (bounded via tanh)
    # Scale by 100 to make typical 0.1% moves map to ~0.1 after tanh
    df['ema9_slope'] = np.tanh(df['ema9'].pct_change(periods=2).fillna(0) * 100)
    df['ema21_slope'] = np.tanh(df['ema21'].pct_change(periods=2).fillna(0) * 100)
    df['ema50_slope'] = np.tanh(df['ema50'].pct_change(periods=2).fillna(0) * 100)
    df['ema100_slope'] = np.tanh(df['ema100'].pct_change(periods=2).fillna(0) * 100)
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


def get_account_state_features(broker_history: list, initial_balance: float, lookback: int) -> np.ndarray:
    """
    Group 6: Account State Features (5 features) - v2 Bounded Design

    All features properly bounded to prevent saturation after many trades.
    Shape: (lookback, 5)

    Features:
    1. equity_growth: tanh((equity / initial_balance) - 1) - Account growth from start
    2. balance_ratio: tanh((balance / initial_balance) - 1) - Free balance growth
    3. unrealized_pnl_pct: unrealized_pnl / used_balance - Current position P&L %
    4. recent_pnl_velocity: tanh(sum(last_10_trades_pnl) / initial_balance) - Recent trading momentum
    5. profit_factor_recent: min(gross_profit_50 / (gross_loss_50 + 1e-6), 5.0) / 5.0 - Win/loss ratio

    Returns:
        np.ndarray: Shape (lookback, 5), dtype float32
    """
    history_slice = broker_history[-lookback:]
    start_idx = lookback - len(history_slice)

    # Vectorized extraction
    equities = np.array([h['equity'] for h in history_slice], dtype=np.float32)
    current_balances = np.array([h['current_balance'] for h in history_slice], dtype=np.float32)
    unrealized_pnls = np.array([h['unrealized_pnl'] for h in history_slice], dtype=np.float32)
    used_balances = np.array([h['used_balance'] for h in history_slice], dtype=np.float32)

    # Pre-allocate features
    features = np.zeros((lookback, 5), dtype=np.float32)

    # Feature 1: Equity growth (bounded via tanh)
    # Maps: 0% → 0.0, +100% → +0.76, -50% → -0.46
    features[start_idx:, 0] = np.tanh((equities / initial_balance) - 1.0)

    # Feature 2: Balance ratio (bounded via tanh)
    features[start_idx:, 1] = np.tanh((current_balances / initial_balance) - 1.0)

    # Feature 3: Unrealized PnL % of position margin
    # Natural ratio: typically ±10% (0.1x leverage), can spike to ±100% with 10x leverage
    unrealized_pct = np.divide(
        unrealized_pnls,
        used_balances,
        out=np.zeros_like(unrealized_pnls),
        where=(used_balances > 0)
    )
    features[start_idx:, 2] = unrealized_pct

    # Feature 4 & 5: Recent PnL velocity and profit factor (computed per timestep from trade history)
    for i, history_state in enumerate(history_slice):
        # Extract all_pnls from the trades in this state
        trades = history_state.get('trades', [])
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        all_pnls = [t['pnl'] for t in closed_trades]

        # Feature 4: Recent PnL velocity (last 10 trades)
        recent_10 = all_pnls[-10:] if len(all_pnls) >= 10 else all_pnls
        recent_pnl_sum = sum(recent_10)
        features[start_idx + i, 3] = np.tanh(recent_pnl_sum / initial_balance)

        # Feature 5: Profit factor (last 50 trades)
        recent_50 = all_pnls[-50:] if len(all_pnls) >= 50 else all_pnls
        gross_profit_50 = sum(pnl for pnl in recent_50 if pnl > 0)
        gross_loss_50 = sum(abs(pnl) for pnl in recent_50 if pnl < 0)
        profit_factor = gross_profit_50 / (gross_loss_50 + 1e-6)
        features[start_idx + i, 4] = min(profit_factor, 5.0) / 5.0  # Cap at 5.0, normalize to [0, 1]

    return features


def get_position_info_features(broker_history: list, lookback: int) -> np.ndarray:
    """
    Group 7: Current Position Info Features (100x Leverage Compatible)

    ...existing docstring...
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

    # Feature 1: Position direction (-1.0=SHORT, 0.0=FLAT, +1.0=LONG)
    features[start_idx:, 0] = np.sign(position_sizes)

    # Feature 2: Leverage used (normalized by 100x max)
    leverage = used_balances / equities
    features[start_idx:, 1] = np.clip(leverage / 100.0, 0.0, 1.0)  # ✅ ADD CLIP

    # Feature 3: Unrealized PnL % of position margin
    unrealized_pct = np.divide(
        unrealized_pnls,
        used_balances,
        out=np.zeros_like(unrealized_pnls),
        where=(used_balances > 0)
    )
    # ✅ FIX: Clip to prevent extreme values with 100x leverage
    features[start_idx:, 2] = np.clip((unrealized_pct * 0.2) + 0.5, 0.0, 1.0)

    # Feature 4: Distance to SL (tanh normalized)
    sl_distance_raw = np.where((sl_prices > 0) & (current_prices > 0),
                               (sl_prices - current_prices) / current_prices,
                               0.0)
    sl_distance = np.where(position_sizes < 0, -sl_distance_raw, sl_distance_raw)
    features[start_idx:, 3] = np.tanh(sl_distance * 10.0)  # Already bounded by tanh

    # Feature 5: Distance to TP (tanh normalized)
    tp_distance_raw = np.where((tp_prices > 0) & (current_prices > 0),
                               (tp_prices - current_prices) / current_prices,
                               0.0)
    tp_distance = np.where(position_sizes < 0, -tp_distance_raw, tp_distance_raw)
    features[start_idx:, 4] = np.tanh(tp_distance * 10.0)  # Already bounded by tanh

    # Feature 6: Risk/Reward ratio
    sl_distance_abs = np.abs(sl_prices - entry_prices)
    tp_distance_abs = np.abs(tp_prices - entry_prices)
    rr_ratio = np.where((sl_distance_abs > 0) & (tp_distance_abs > 0),
                        tp_distance_abs / sl_distance_abs,
                        0.0)
    features[start_idx:, 5] = np.clip(rr_ratio / 10.0, 0, 1)  # Already clipped ✅

    # Feature 7: Position duration
    entry_bars = np.array([h.get('entry_bar', 0) if h['position_size'] != 0 else 0 for h in history_slice], dtype=np.float32)
    duration = np.where(position_sizes != 0, steps - entry_bars, 0)
    # ✅ FIX: Clip after tanh (defensive programming)
    features[start_idx:, 6] = np.clip(np.tanh(duration / float(lookback)), -1.0, 1.0)

    return features


def get_vp_bins_features(vp_obj, lookback: int) -> np.ndarray:
    """
    Extract VP bins (volume distribution histogram) from EnhancedVolumeProfile.

    Direct access to VP internal state for feature extraction.

    Returns:
        np.ndarray: Shape (lookback, n_bins), range [0, 1]
                   Volume distribution already normalized by VP class.

    Processing:
        - Extracts from circular buffer (daily_bins_history)
        - Handles cases: empty, partial fill, full circular buffer
        - Already normalized to [0, 1] when stored
    """
    # Handle empty case
    if vp_obj.daily_bins_count == 0:
        return np.zeros((lookback, vp_obj.n_bins), dtype=np.float32)

    # Handle partial fill (less data than requested)
    if vp_obj.daily_bins_count < lookback:
        result = np.zeros((lookback, vp_obj.n_bins), dtype=np.float32)
        # Place available data at the end (most recent)
        available_data = vp_obj.daily_bins_history[:vp_obj.daily_bins_count].cpu().numpy()
        result[-vp_obj.daily_bins_count:] = available_data
        return result

    # Handle full circular buffer
    idx = vp_obj.daily_bins_idx
    if idx >= lookback:
        # Simple slice (no wraparound)
        bins_tensor = vp_obj.daily_bins_history[idx - lookback:idx]
    else:
        # Wraparound case: concatenate end + beginning
        part1 = vp_obj.daily_bins_history[vp_obj.lookback_window - (lookback - idx):]
        part2 = vp_obj.daily_bins_history[:idx]
        bins_tensor = torch.cat([part1, part2], dim=0)

    return bins_tensor.cpu().numpy()


def get_vp_bins_features_visible(price_data: pd.DataFrame, current_step: int,
                                 lookback: int, n_bins: int = 50) -> np.ndarray:
    """
    Calculate ROLLING CUMULATIVE Volume Profile for VISIBLE RANGE.

    Each timestep shows cumulative volume UP TO that point, so the agent
    can see how volume built up over time at each price level.
    This provides temporal context - the CNN can learn volume momentum patterns.

    Args:
        price_data: DataFrame with OHLC + volume columns
        current_step: Current step index in the data
        lookback: Number of timesteps (288)
        n_bins: Number of price bins (default 50)

    Returns:
        np.ndarray: Shape (lookback, n_bins), range [0, 1]
                   Rolling cumulative VP showing volume accumulation over time
    """
    # Get visible window data
    start_idx = max(0, current_step - lookback)
    window_data = price_data.iloc[start_idx:current_step]

    if len(window_data) == 0:
        return np.zeros((lookback, n_bins), dtype=np.float32)

    # Calculate visible range from FULL window (for consistent bin edges)
    visible_low = float(window_data['low'].min())
    visible_high = float(window_data['high'].max())

    if visible_high <= visible_low:
        return np.zeros((lookback, n_bins), dtype=np.float32)

    bin_size = (visible_high - visible_low) / n_bins

    # Calculate ROLLING cumulative VP (vectorized where possible)
    vp_bins_temporal = np.zeros((len(window_data), n_bins), dtype=np.float32)
    cumulative_vp = np.zeros(n_bins, dtype=np.float32)

    # Build VP progressively - each row shows volume accumulated up to that point
    for i, (_, row) in enumerate(window_data.iterrows()):
        price = float(row['close'])
        volume = float(row['volume'])

        # Find which bin this price falls into
        bin_idx = int((price - visible_low) / bin_size)
        bin_idx = max(0, min(bin_idx, n_bins - 1))

        # Add volume to cumulative VP
        cumulative_vp[bin_idx] += volume

        # Store cumulative state at this timestep
        vp_bins_temporal[i] = cumulative_vp.copy()

    # Normalize each timestep independently to [0, 1]
    # This preserves temporal patterns while keeping values in valid range
    for i in range(len(window_data)):
        max_vol = vp_bins_temporal[i].max()
        if max_vol > 0:
            vp_bins_temporal[i] /= max_vol

    # Pad if needed (beginning of episode)
    if len(window_data) < lookback:
        padding = np.zeros((lookback - len(window_data), n_bins), dtype=np.float32)
        vp_bins_temporal = np.vstack([padding, vp_bins_temporal])

    return vp_bins_temporal


def get_vp_levels_features_visible(price_data: pd.DataFrame, current_step: int,
                                   lookback: int, n_bins: int = 50) -> np.ndarray:
    """
    Extract VP levels features for VISIBLE RANGE (vectorized, OHLC-aware).

    Args:
        price_data: DataFrame with OHLC + volume columns
        current_step: Current step index in the data
        lookback: Number of timesteps (288)
        n_bins: Number of price bins (default 50)

    Returns:
        np.ndarray: Shape (lookback, 26), range [-1, 1] for continuous, {0, 1} for binary

        Features (per timestep):
        === Continuous Features (20): OHLC distances to 5 levels ===
        0-3:   Day High distances  [open, high, low, close]
        4-7:   VAH distances       [open, high, low, close]
        8-11:  POC distances       [open, high, low, close]
        12-15: VAL distances       [open, high, low, close]
        16-19: Day Low distances   [open, high, low, close]

        === Binary Features (6): Spatial relationships with wicks ===
        20: close_in_va         # 1 if close in Value Area, else 0
        21: close_above_va      # 1 if close > VAH, else 0
        22: close_below_va      # 1 if close < VAL, else 0
        23: wick_touched_va     # 1 if high >= VAH OR low <= VAL, else 0
        24: close_above_poc     # 1 if close > POC, else 0
        25: wick_crossed_poc    # 1 if low < POC < high, else 0
    """
    from features.visible_range_vp import VisibleRangeVP

    # Get visible window data
    start_idx = max(0, current_step - lookback)
    window_data = price_data.iloc[start_idx:current_step]

    if len(window_data) == 0:
        return np.zeros((lookback, 26), dtype=np.float32)

    # Calculate VP for visible range to get levels
    vp = VisibleRangeVP(n_bins=n_bins)
    _, levels = vp.calculate_vp(window_data)

    # Extract OHLC prices for vectorized calculations
    open_prices = window_data['open'].values.astype(np.float32)
    high_prices = window_data['high'].values.astype(np.float32)
    low_prices = window_data['low'].values.astype(np.float32)
    close_prices = window_data['close'].values.astype(np.float32)

    # Handle padding for beginning of episode
    if len(close_prices) < lookback:
        pad_size = lookback - len(close_prices)
        padding_value = close_prices[0] if len(close_prices) > 0 else 0.0
        open_prices = np.concatenate([np.full(pad_size, padding_value), open_prices])
        high_prices = np.concatenate([np.full(pad_size, padding_value), high_prices])
        low_prices = np.concatenate([np.full(pad_size, padding_value), low_prices])
        close_prices = np.concatenate([np.full(pad_size, padding_value), close_prices])

    # Get levels
    visible_high = levels['high']
    vah = levels['vah']
    poc = levels['poc']
    val = levels['val']
    visible_low = levels['low']

    # Calculate range for normalization
    price_range = visible_high - visible_low
    if price_range < 1e-6:
        # Flat market - no meaningful levels
        return np.zeros((lookback, 26), dtype=np.float32)

    # === CONTINUOUS FEATURES (20): OHLC distances to each level ===

    # Level 1: Day High (features 0-3)
    open_to_high = np.clip((visible_high - open_prices) / price_range, -1, 1)
    high_to_high = np.clip((visible_high - high_prices) / price_range, -1, 1)
    low_to_high = np.clip((visible_high - low_prices) / price_range, -1, 1)
    close_to_high = np.clip((visible_high - close_prices) / price_range, -1, 1)

    # Level 2: VAH (features 4-7)
    open_to_vah = np.clip((vah - open_prices) / price_range, -1, 1)
    high_to_vah = np.clip((vah - high_prices) / price_range, -1, 1)
    low_to_vah = np.clip((vah - low_prices) / price_range, -1, 1)
    close_to_vah = np.clip((vah - close_prices) / price_range, -1, 1)

    # Level 3: POC (features 8-11)
    open_to_poc = np.clip((poc - open_prices) / price_range, -1, 1)
    high_to_poc = np.clip((poc - high_prices) / price_range, -1, 1)
    low_to_poc = np.clip((poc - low_prices) / price_range, -1, 1)
    close_to_poc = np.clip((poc - close_prices) / price_range, -1, 1)

    # Level 4: VAL (features 12-15)
    open_to_val = np.clip((val - open_prices) / price_range, -1, 1)
    high_to_val = np.clip((val - high_prices) / price_range, -1, 1)
    low_to_val = np.clip((val - low_prices) / price_range, -1, 1)
    close_to_val = np.clip((val - close_prices) / price_range, -1, 1)

    # Level 5: Day Low (features 16-19) - inverted (positive when above low)
    open_to_low = np.clip((open_prices - visible_low) / price_range, -1, 1)
    high_to_low = np.clip((high_prices - visible_low) / price_range, -1, 1)
    low_to_low = np.clip((low_prices - visible_low) / price_range, -1, 1)
    close_to_low = np.clip((close_prices - visible_low) / price_range, -1, 1)

    # === BINARY FEATURES (6): Spatial relationships ===

    # Value Area relationship (features 20-23)
    close_in_va = ((close_prices >= val) & (close_prices <= vah)).astype(np.float32)
    close_above_va = (close_prices > vah).astype(np.float32)
    close_below_va = (close_prices < val).astype(np.float32)
    wick_touched_va = ((high_prices >= vah) | (low_prices <= val)).astype(np.float32)

    # POC relationship (features 24-25)
    close_above_poc = (close_prices > poc).astype(np.float32)
    wick_crossed_poc = ((low_prices < poc) & (poc < high_prices)).astype(np.float32)

    # === Stack all 26 features ===
    features = np.stack([
        # Day High (0-3)
        open_to_high, high_to_high, low_to_high, close_to_high,
        # VAH (4-7)
        open_to_vah, high_to_vah, low_to_vah, close_to_vah,
        # POC (8-11)
        open_to_poc, high_to_poc, low_to_poc, close_to_poc,
        # VAL (12-15)
        open_to_val, high_to_val, low_to_val, close_to_val,
        # Day Low (16-19)
        open_to_low, high_to_low, low_to_low, close_to_low,
        # Binary features (20-25)
        close_in_va, close_above_va, close_below_va, wick_touched_va,
        close_above_poc, wick_crossed_poc
    ], axis=1)

    return features


def get_vp_levels_features(vp_obj, lookback: int, price_series: np.ndarray) -> np.ndarray:
    """
    Extract VP levels with full context (VECTORIZED).

    Direct access to VP internal state for feature extraction.

    Args:
        vp_obj: EnhancedVolumeProfile instance
        lookback: Number of timesteps
        price_series: Close prices for lookback window [lookback]

    Returns:
        np.ndarray: Shape (lookback, 9), range [-1, 1] or {0, 1} for binary

        Features (per timestep, ordered by price level HIGH→LOW):
        === Continuous Features (normalized distances) ===
        0. price_to_day_high: Distance from price to session high
        1. price_to_vah: Distance from price to VAH
        2. price_to_poc: Distance from price to POC
        3. price_to_val: Distance from price to VAL
        4. price_to_day_low: Distance from price to session low

        === Binary/Categorical Features (spatial relationships) ===
        5. in_value_area: 1 if price between VAL and VAH, else 0
        6. above_poc: 1 if price > POC, else 0
        7. session_intersection: Today's VA vs yesterday's VA position
           -1.0: Below (today's VA completely below yesterday's)
           -0.5: Inside+Below (extends down with overlap)
            0.0: Inside or Above+Below (consolidation/wide expansion)
           +0.5: Inside+Above (extends up with overlap)
           +1.0: Above (today's VA completely above yesterday's)
        8. poc_crossover: 1 if price crossed POC in last 5 bars, else 0
    """
    # Handle no VP data case
    if vp_obj.current_day_vah is None or vp_obj.current_day_high is None:
        return np.zeros((lookback, 9), dtype=np.float32)

    # Current session levels
    day_high = vp_obj.current_day_high
    vah = vp_obj.current_day_vah
    poc = vp_obj.current_day_poc
    val = vp_obj.current_day_val
    day_low = vp_obj.current_day_low

    # Previous day levels (if available)
    prev_vah = None
    prev_val = None
    if len(vp_obj.daily_sessions) > 0:
        prev_session = vp_obj.daily_sessions[-1]
        prev_vah = prev_session.get('vah')
        prev_val = prev_session.get('val')

    # Calculate range for normalization
    price_range = day_high - day_low
    if price_range < 1e-6:
        # Flat market - no meaningful levels
        return np.zeros((lookback, 9), dtype=np.float32)

    # === VECTORIZED: Calculate all distances at once ===
    # Shape: [lookback]
    prices = price_series.astype(np.float32)

    # Features 0-4: Distances to levels (normalized by day range)
    # Positive = level above price, Negative = level below price
    dist_to_high = (day_high - prices) / price_range
    dist_to_vah = (vah - prices) / price_range
    dist_to_poc = (poc - prices) / price_range
    dist_to_val = (val - prices) / price_range
    dist_to_low = (prices - day_low) / price_range  # Inverted: positive when above low

    # Clip to [-1, 1]
    dist_to_high = np.clip(dist_to_high, -1, 1)
    dist_to_vah = np.clip(dist_to_vah, -1, 1)
    dist_to_poc = np.clip(dist_to_poc, -1, 1)
    dist_to_val = np.clip(dist_to_val, -1, 1)
    dist_to_low = np.clip(dist_to_low, -1, 1)

    # === Feature 5: In Value Area (vectorized) ===
    # Shape: [lookback]
    in_va = ((prices >= val) & (prices <= vah)).astype(np.float32)

    # === Feature 6: Above POC (vectorized) ===
    above_poc = (prices > poc).astype(np.float32)

    # === Feature 7: Session Intersection (scalar, broadcast to all timesteps) ===
    # 6 possible cases for today's VA vs yesterday's VA:
    # 1. Inside: Today's VA completely inside yesterday's VA
    # 2. Above: Today's VA completely above yesterday's VA
    # 3. Below: Today's VA completely below yesterday's VA
    # 4. Inside+Below: Today's VA extends below yesterday's but overlaps
    # 5. Inside+Above: Today's VA extends above yesterday's but overlaps
    # 6. Above+Below: Today's VA encompasses yesterday's VA (wider range)

    session_intersection = 0.0
    if prev_vah is not None and prev_val is not None:
        # Case 1: Inside - Today's VA completely inside yesterday's
        if val >= prev_val and vah <= prev_vah:
            session_intersection = 0.0  # Consolidation/inside bar

        # Case 2: Above - Today's VA completely above yesterday's
        elif val > prev_vah:
            session_intersection = 1.0  # Bullish expansion

        # Case 3: Below - Today's VA completely below yesterday's
        elif vah < prev_val:
            session_intersection = -1.0  # Bearish expansion

        # Case 4: Inside+Below - Today's VA extends below but overlaps
        elif val < prev_val and vah >= prev_val and vah <= prev_vah:
            session_intersection = -0.5  # Partial bearish overlap

        # Case 5: Inside+Above - Today's VA extends above but overlaps
        elif vah > prev_vah and val >= prev_val and val <= prev_vah:
            session_intersection = 0.5  # Partial bullish overlap

        # Case 6: Above+Below - Today's VA encompasses yesterday's
        elif val < prev_val and vah > prev_vah:
            session_intersection = 0.0  # Wide expansion (neutral)

    # Broadcast to all timesteps
    session_intersection_arr = np.full(lookback, session_intersection, dtype=np.float32)

    # === Feature 8: POC Crossover (vectorized sliding window) ===
    # Check if POC is between min/max of last 5 bars
    poc_crossover = np.zeros(lookback, dtype=np.float32)

    if lookback >= 6:
        # Use numpy stride tricks for sliding window (efficient!)
        from numpy.lib.stride_tricks import sliding_window_view

        # Create sliding windows of size 6 (current + 5 previous)
        windows = sliding_window_view(prices, window_shape=6)

        # Check if POC is within each window's range
        window_mins = windows.min(axis=1)
        window_maxs = windows.max(axis=1)
        poc_in_range = ((poc >= window_mins) & (poc <= window_maxs)).astype(np.float32)

        # Place results at correct indices (starting from index 5)
        poc_crossover[5:] = poc_in_range

    # === Stack all features ===
    # Shape: [lookback, 9]
    features = np.stack([
        dist_to_high,              # 0: Distance to day high
        dist_to_vah,               # 1: Distance to VAH
        dist_to_poc,               # 2: Distance to POC
        dist_to_val,               # 3: Distance to VAL
        dist_to_low,               # 4: Distance to day low
        in_va,                     # 5: In value area (binary)
        above_poc,                 # 6: Above POC (binary)
        session_intersection_arr,  # 7: Session intersection (categorical)
        poc_crossover              # 8: POC crossover (binary)
    ], axis=1)

    return features
