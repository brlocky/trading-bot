"""
Normalization Configuration Helper

Standalone helper to create feature normalization configurations
without depending on config files or external dependencies.
"""


# Environment configuration helpers
def get_default_environment_config():
    """Get default trading environment configuration optimized for stability"""
    return {
        'initial_balance': 100000.0,
        'buy_threshold': 0.5,         # 🛡️ STABLE: 0.3 → 0.5 (more conservative thresholds)
        'sell_threshold': -0.5,       # 🛡️ STABLE: -0.3 → -0.5 (more conservative thresholds)
    }


def get_model_config():
    """Get training configuration for RecurrentPPO - FIXED VALUE FUNCTION"""
    return {
        'total_timesteps': 100_000,        # Total training steps
        'window_size': 96,                # Input window (96x15m = 1 day)

        'n_steps': 2048,                     # Steps per rollout (was 2048)

        # PPO hyperparameters
        # Learning rate is very low (3e-5) — that’s fine for stability, but training will be slow.
        # Could gradually increase once KL stabilizes.
        'learning_rate': 3e-4,            # Lower LR for smoother updates (was 3e-4)

        'batch_size_gpu': 128,            # GPU batch size for training
        'batch_size_cpu': 128,            # CPU batch size for rollout collection
        'n_epochs': 4,                   # Gradient passes per PPO update
        'gamma': 0.99,                   # Discount factor, values future rewards more
        'gae_lambda': 0.95,               # Smoothing factor for advantage estimation

        # KL spikes still happen occasionally (0.13–0.44) — small but noticeable.
        # This is normal with financial data;
        # can reduce clip_range even further to 0.05–0.08 if needed.
        'clip_range': 0.2,                # PPO clipping for stable updates

        # Value function tuning
        'clip_range_vf': None,            # Clip value function changes
        'normalize_advantage': True,      # Normalize advantages for stability
        'target_kl': 0.2,                 # Allow slightly larger KL before early stop (was 0.3)
        'stats_window_size': 100,         # Moving window for stats tracking
        'seed': 42,                       # Random seed for reproducibility

        # Entropy loss around -3 — enough exploration; you don’t need to increase ent_coef.
        'ent_coef': 0.01,               # Better exploration using 0.01

        # Value loss sometimes rises — indicates the value network still struggles; could increase vf_coef slightly (0.9).
        'vf_coef': 0.5,                   # Increase weight on value loss to improve value fitting (was 0.6)
        'max_grad_norm': 0.5,             # Relax gradient clipping slightly (was 0.5)
        'use_sde': False,                  # Use State-Dependent Exploration (stable noise)
        'sde_sample_freq': -1,            # Resample SDE noise only once per rollout

        'n_envs': 4,                      # Parallel environments for efficiency

        # Network architecture
        'hidden_layers_vf': [64, 64],        # Value function network depth
        'hidden_layers_pi': [64, 64],        # Policy network depth
        'lstm_hidden_size': 64,               # LSTM hidden layer size for memory
        'lstm_num_layers': 1,                  # Number of LSTM layers for memory
        'activation_function': 'ReLU',         # Activation function for all layers
        'ortho_init': True,                    # Orthogonal weight initialization for stability
    }
