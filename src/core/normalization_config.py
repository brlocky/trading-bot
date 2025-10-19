"""
Normalization Configuration Helper

Standalone helper to create feature normalization configurations
without depending on config files or external dependencies.
"""

from typing import Dict, List


def get_features_config() -> Dict[str, List[str]]:
    return {
        '15m': [
            'ema20', 'ema200', 'vwap', 'bb_position',
            'rsi',  'macd', 'macd_hist', 'volume_ma20', 'volume_ratio', 'volatility',
            'ema9_ema21_cross', 'ema20_ema50_cross'
        ],
        '1h': [
            'ema20', 'ema50', 'ema200', 'rsi', 'macd', 'macd_hist'
        ],
        'D': [
            'ema200', 'macd', 'macd_hist', 'rsi'
        ],
        'W': [
            'ema50', 'macd', 'macd_hist', 'rsi'
        ]
    }


def get_features_list() -> Dict[str, str]:
    """
    Get the complete mapping of features to their normalizers.
    Combines base OHLCV features with timeframe indicators.
    """
    # Feature type to normalizer mapping
    feature_normalizers = {
        # Base OHLCV
        'open': 'price_ratio',
        'high': 'price_ratio',
        'low': 'price_ratio',
        'close': 'price_baseline',
        'volume': 'volume_log',

        # Technical indicators from all features (including commented ones)
        'ema9': 'price_ratio',
        'ema21': 'price_ratio',
        'ema50': 'price_ratio',
        'ema20': 'price_ratio',
        'ema200': 'price_ratio',
        'vwap': 'price_ratio',
        'bb_upper': 'price_ratio',
        'bb_lower': 'price_ratio',
        'rsi': 'bounded_0_100',
        'bb_position': 'bounded_0_1',
        'macd': 'oscillator',
        'macd_hist': 'oscillator',
        'volume_ma20': 'volume_log',
        'volume_ratio': 'ratio',
        'volatility': 'standard',
        'ema9_ema21_cross': 'bounded_0_1',
        'ema20_ema50_cross': 'bounded_0_1'
    }

    # Start with base OHLCV features
    features = {
        'open': feature_normalizers['open'],
        'high': feature_normalizers['high'],
        'low': feature_normalizers['low'],
        'close': feature_normalizers['close'],
        'volume': feature_normalizers['volume']
    }

    # Get timeframe indicators
    timeframe_indicators = get_features_config()

    # Add 15m indicators directly
    if '15m' in timeframe_indicators:
        for indicator in timeframe_indicators['15m']:
            normalizer = feature_normalizers.get(indicator)
            if not normalizer:
                raise ValueError(f"No normalizer found for indicator: {indicator}")
            features[indicator] = normalizer

    # Add other timeframes with suffix
    for timeframe, indicators in timeframe_indicators.items():
        if timeframe != '15m':  # Skip 15m as it's already added
            for indicator in indicators:
                feature_name = f"{indicator}_{timeframe}"
                normalizer = feature_normalizers.get(indicator)
                if not normalizer:
                    raise ValueError(f"No normalizer found for indicator: {indicator}")
                features[feature_name] = normalizer

    return features


# Environment configuration helpers
def get_default_environment_config():
    """Get default trading environment configuration optimized for stability"""
    return {
        'initial_balance': 1000.0,
        'buy_threshold': 0.5,         # 🛡️ STABLE: 0.3 → 0.5 (more conservative thresholds)
        'sell_threshold': -0.5,       # 🛡️ STABLE: -0.3 → -0.5 (more conservative thresholds)
    }


def get_model_config():
    """Get training configuration for RecurrentPPO - FIXED VALUE FUNCTION"""
    return {
        # PPO training configuration
        'total_timesteps': 200000,
        'n_eval_episodes': 3,
        'train_test_split': 0.8,

        # Environment configuration
        'window_size': 336,

        # PPO hyperparameters - DUAL LEARNING RATES
        'learning_rate': lambda f: 3e-4 * f,  # Linearly decay LR
        'n_steps': 2048,                  # 🔧 INCREASED: 1024 → 2048 (longer rollouts)
        'batch_size_gpu': 256,            # Keep large batches
        'batch_size_cpu': 128,
        'n_epochs': 4,                    # 🔧 REDUCED: Was unstable with 10
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.1,     # ← ADD THIS: More conservative updates

        # Additional hyperparameters - VALUE FUNCTION FOCUS
        'clip_range_vf': None,            # 🔧 DISABLED: Let VF learn freely
        'normalize_advantage': True,
        'target_kl': 0.03,  # stop epoch if KL > 0.03
        'stats_window_size': 25,
        'seed': 42,

        'ent_coef': 0.01,                 # 🔧 REDUCED: 0.02 → 0.01 (less randomness)
        'vf_coef': 1,                   # 🔧 Better value function learning
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,

        # Environment configuration
        'n_envs': 4,                      # 🔧 REDUCED: 8 → 4 (higher quality rollouts)

        # Network architecture
        'hidden_layers_vf': [256, 256],
        'hidden_layers_pi': [256, 128],
        'lstm_hidden_size': 128,
        'activation_function': 'tanh',
        'ortho_init': True,
    }
