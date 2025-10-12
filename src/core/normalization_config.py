"""
Normalization Configuration Helper

Standalone helper to create feature normalization configurations
without depending on config files or external dependencies.
"""

from typing import Dict, List


def get_default_feature_normalization() -> Dict[str, str]:
    """
    Get default feature normalization configuration.
    Maps feature names to normalizer types based on actual production features.
    """
    return {
        # === BASE FEATURES ===
        # OHLCV
        'open': 'price_ratio',
        'high': 'price_ratio',
        'low': 'price_ratio',
        'close': 'price_baseline',
        'volume': 'volume_log',

        # === 15M TECHNICAL INDICATORS ===
        'ema20': 'price_ratio',
        'ema50': 'price_ratio',
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
        'ema9_ema21_cross': 'bounded_0_1',  # binary 0/1 mapped to [-1,+1]
        'ema20_ema50_cross': 'bounded_0_1',  # binary 0/1 mapped to [-1,+1]

        # === INTERPOLATED 1H FEATURES ===
        'ema20_1h': 'price_ratio',
        'ema50_1h': 'price_ratio',
        'ema200_1h': 'price_ratio',
        'rsi_1h': 'bounded_0_100',
        'macd_1h': 'oscillator',
        'macd_hist_1h': 'oscillator',

        # === INTERPOLATED DAILY FEATURES ===
        'ema20_D': 'price_ratio',
        'ema50_D': 'price_ratio',
        'macd_hist_D': 'oscillator',
        'rsi_D': 'bounded_0_100',

        # === INTERPOLATED WEEKLY FEATURES ===
        'ema20_W': 'price_ratio',
        'ema50_W': 'price_ratio',
        'macd_hist_W': 'oscillator',
        'rsi_W': 'bounded_0_100',

        # === INTERPOLATED MONTHLY FEATURES ===
        'ema20_M': 'price_ratio',
        'ema50_M': 'price_ratio',
        'macd_hist_M': 'oscillator',
        'rsi_M': 'bounded_0_100',
    }


# Environment configuration helpers
def get_default_environment_config():
    """Get default trading environment configuration"""
    return {
        'initial_balance': 1000000.0,
        'window_size': 672,
    }


def get_training_config():
    """Get default training configuration matching rl_config.ini"""
    return {
        # PPO training configuration
        'total_timesteps': 10000,  # Keep current training length
        'early_stop_patience': 8,  # Increased patience for slower learning
        'early_stop_threshold': 0.01,
        'eval_freq': 1000,  # Less frequent evaluation
        'n_eval_episodes': 10,
        'train_test_split': 0.8,

        # PPO hyperparameters
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size_gpu': 64,
        'batch_size_cpu': 128,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.1,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'normalize_advantage': True,
        'use_sde': True,
        'sde_sample_freq': 4,

        # Network architecture
        'hidden_layers_pi': [512, 256, 128],
        'hidden_layers_vf': [512, 256, 128],
        'activation_function': 'tanh',
        'ortho_init': False,
    }


def get_timeframe_indicators() -> Dict[str, List[str]]:
    """Get timeframe indicators configuration matching actual TIMEFRAME_INDICATORS"""
    return {
        '15m': [
            'ema20', 'ema50', 'ema200', 'vwap', 'bb_upper', 'bb_lower',
            'rsi', 'bb_position', 'macd', 'macd_hist', 'volume_ma20', 'volume_ratio', 'volatility',
            'ema9_ema21_cross', 'ema20_ema50_cross'
        ],
        '1h': [
            'ema20', 'ema50', 'ema200', 'rsi', 'macd', 'macd_hist'
        ],
        'D': [
            'ema20', 'ema50', 'macd_hist', 'rsi'
        ],
        'W': [
            'ema20', 'ema50', 'macd_hist', 'rsi'
        ],
        'M': [
            'ema20', 'ema50', 'macd_hist', 'rsi'
        ]
    }
