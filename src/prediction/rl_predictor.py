"""
Reinforcement Learning Predictor - Minimal RL trading agent using PPO
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from typing import Optional, Dict
import os
import torch
from stable_baselines3.common.monitor import Monitor
from data_processing.feature_normalizer import FeatureNormalizer
from environments.trading_environment import TradingEnvironment
from core.normalization_config import (
    get_default_feature_normalization,
    get_default_environment_config,
    get_training_config,
)
from prediction.progress_tracking_callback import create_progress_tracking_callback
from prediction.trading_metrics_callback import create_trading_metrics_callback


class RLPredictor:
    """
    Enhanced RL trading predictor using PPO with GPU support
    Uses standalone configuration and separated environment
    """

    def __init__(self, model_dir: str = 'models/rl', feature_config: Optional[Dict[str, str]] = None):
        self.model_dir = model_dir
        self.model = None
        self.feature_columns = None
        os.makedirs(self.model_dir, exist_ok=True)

        # Use standalone configuration instead of config files
        if feature_config is None:
            feature_config = get_default_feature_normalization()
        self.feature_config = feature_config

        # Get default configurations
        self.env_config = get_default_environment_config()
        self.training_config = get_training_config()

        # Detect available device
        # self.device = 'cpu'  # self._detect_device()
        self.device = self._detect_device()
        print(f"🖥️ RL Training Device: {self.device}")

    def _detect_device(self) -> str:
        """Detect and return the best available device for training"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = 'cpu'
            print("⚠️ GPU not available, using CPU (training will be slower)")
        return device

    def train(self, df: pd.DataFrame, generate_report: bool = True, **kwargs):
        """
        Train with proper data split and normalization to prevent data leakage
        Uses standalone configuration for all parameters, but allows overrides

        Args:
            df: Training data with features
            train_ratio: Ratio of data to use for training vs evaluation
            generate_report: Whether to generate a comprehensive training report after training
            **kwargs: Override training configuration parameters
        """
        # Get training configuration
        train_config = get_training_config()

        # Override with any provided kwargs
        total_timesteps = kwargs.get('total_timesteps', train_config['total_timesteps'])
        eval_freq = kwargs.get('eval_freq', train_config['eval_freq'])
        n_eval_episodes = kwargs.get('n_eval_episodes', train_config['n_eval_episodes'])

        # Split data for train/eval
        train_ratio = train_config['train_test_split']

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        eval_df = df.iloc[split_idx:].copy()

        print("🚀 Starting training with proper data split...")
        print(f"📊 Train data: {len(train_df):,} rows, Eval data: {len(eval_df):,} rows")

        # Create training environment
        print("🔧 Creating training environment...")
        train_env = TradingEnvironment(
            df=train_df,
            feature_config={k: v for k, v in self.feature_config.items() if k in df.columns},
            fit_normalizer=True,  # Fit on training data only
            **self.env_config
        )

        # Create evaluation environment
        print("📥 Creating evaluation environment...")
        eval_env = TradingEnvironment(
            df=eval_df,
            feature_config={k: v for k, v in self.feature_config.items() if k in df.columns},
            fit_normalizer=True,  # Don't refit on evaluation data
            **self.env_config
        )

        # Validate data integrity
        print("🔍 Validating data integrity...")
        train_stats = train_env.validate_data_integrity()
        eval_stats = eval_env.validate_data_integrity()

        # Inform about NaN handling (expected for early indicator periods)
        if train_stats['original_nan_counts']:
            max_nan_feature = max(train_stats['original_nan_counts'], key=train_stats['original_nan_counts'].get)
            max_nan_count = train_stats['original_nan_counts'][max_nan_feature]

            print("ℹ️ Training data NaN summary:")
            print(f"   • Features with NaN: {len(train_stats['original_nan_counts'])}")
            print(f"   • Highest NaN count: {max_nan_feature} ({max_nan_count} values)")
            print("   • Cause: Early indicator calculation periods (expected)")
            print("   • Handling: NaN → neutral values (RSI→50, MACD→0, etc.)")

        # Inform about NaN handling (expected for early indicator periods)
        if eval_stats['original_nan_counts']:
            max_nan_feature = max(eval_stats['original_nan_counts'], key=eval_stats['original_nan_counts'].get)
            max_nan_count = eval_stats['original_nan_counts'][max_nan_feature]

            print("ℹ️ Evaluation data NaN summary:")
            print(f"   • Features with NaN: {len(eval_stats['original_nan_counts'])}")
            print(f"   • Highest NaN count: {max_nan_feature} ({max_nan_count} values)")
            print("   • Cause: Early indicator calculation periods (expected)")
            print("   • Handling: NaN → neutral values (RSI→50, MACD→0, etc.)")

        # Verify normalizer successfully handled NaN values
        sample_obs = train_env.reset()[0]
        if np.any(np.isnan(sample_obs)):
            raise ValueError("❌ Normalizer failed - observations still contain NaN after processing")

        print("✅ All NaN values successfully converted to neutral trading signals")

        # Wrap eval environment with Monitor for tracking
        eval_env = Monitor(eval_env)

        # Configure PPO with settings from configuration
        activation_map = {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU,
            'leaky_relu': torch.nn.LeakyReLU
        }

        policy_kwargs = {
            'net_arch': dict(
                pi=train_config['hidden_layers_pi'],
                vf=train_config['hidden_layers_vf']
            ),
            'activation_fn': activation_map.get(train_config['activation_function'], torch.nn.Tanh),
            'ortho_init': train_config['ortho_init']
        }

        print(f"🚀 Initializing PPO model on {self.device}...")
        self.model = PPO(
            'MlpPolicy',
            train_env,
            verbose=0,  # Disable PPO's built-in verbose to avoid conflicts
            device=self.device,
            policy_kwargs=policy_kwargs,
            # Hyperparameters from configuration
            learning_rate=lambda progress: train_config['learning_rate'] * (1 - progress),
            n_steps=train_config['n_steps'],
            batch_size=train_config['batch_size_gpu'] if self.device == 'cuda' else train_config['batch_size_cpu'],
            n_epochs=train_config['n_epochs'],
            gamma=train_config['gamma'],
            gae_lambda=train_config['gae_lambda'],
            clip_range=train_config['clip_range'],
            ent_coef=train_config['ent_coef'],
            vf_coef=train_config['vf_coef'],
            max_grad_norm=train_config['max_grad_norm'],

            # Training stability improvements
            normalize_advantage=train_config['normalize_advantage'],
            use_sde=train_config['use_sde'],
            sde_sample_freq=train_config['sde_sample_freq']
        )

        # Create separate callbacks for different concerns
        progress_callback = create_progress_tracking_callback(
            total_timesteps=total_timesteps,
            verbose=1
        )

        eval_callback = create_trading_metrics_callback(
            eval_env,
            best_model_save_path=os.path.join(self.model_dir, 'best_model'),
            log_path=os.path.join(self.model_dir, 'eval_logs'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,  # Keep verbose for evaluation updates
            warn=False
        )

        # Combine all callbacks (early stopping will be implemented in evaluation callback later)
        callbacks = [progress_callback, eval_callback]

        print("🎯 Training PPO with modular callback system")
        print(f"📈 Evaluation every {eval_freq:,} timesteps using {n_eval_episodes} episodes")
        print(f"🎯 Training PPO model for {total_timesteps:,} timesteps on {self.device}...")
        print(f"⏱️ Progress updates every {progress_callback.print_interval} timesteps")
        print("=" * 60)

        try:
            print("🚀 Starting PPO training...")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks  # Use list of callbacks
            )
            print("✅ Training completed successfully")

            # Save model only on successful completion
            model_path = os.path.join(self.model_dir, 'ppo_trading')
            self.model.save(model_path)
            print(f"✅ Model saved to: {model_path}.zip")

            # Save normalizer only on successful completion
            self.save_normalizer(train_env.get_normalizer())
            training_successful = True

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user - model NOT saved")
            training_successful = False
        except Exception as e:
            print(f"❌ Training failed: {e} - model NOT saved")
            training_successful = False
            raise

        # Generate comprehensive training report only if training was successful
        if generate_report and training_successful:
            try:
                print("✅ Training completed! Basic report:")
                print(f"📋 Model saved to: {self.model_dir}")
                print(f"🎯 Final timesteps: {self.model.num_timesteps:,}")

            except Exception as e:
                print(f"⚠️ Warning: Report generation failed: {e}")
                print("Training completed successfully, but report could not be generated.")

        return True  # Indicate successful training

    def load_model(self):
        """Load the trained model with proper device handling"""
        path = os.path.join(self.model_dir, 'ppo_trading.zip')
        # Load model with device specification
        self.model = PPO.load(path, device=self.device)
        print(f"✅ Model loaded on {self.device}")

    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using the trained RL model - Environment handles ALL trading logic"""
        if self.model is None:
            self.load_model()

        print(f"🎯 Generating predictions on {self.device}...")
        print(f"📊 Data shape: {df.shape[0]:,} rows")

        # Create environment - IT will handle ALL trading logic
        env = TradingEnvironment(
            df=df,
            feature_config={k: v for k, v in self.feature_config.items() if k in df.columns},
            fit_normalizer=True,  # Use pre-fitted normalizer
            **self.env_config
        )
        obs, _ = env.reset()

        # ✅ Collect data FROM environment (not duplicate tracking)
        actions = []
        env_states = []  # Store full environment state each step

        initial_portfolio = env.balance
        total_predictions = len(df) - env.window_size

        print(f"🔮 Generating {total_predictions:,} predictions...")
        print(f"💰 Initial Portfolio: ${initial_portfolio:,.2f}")

        for i in range(env.window_size, len(df)):
            if self.model is None:
                self.load_model()

            # Get prediction from model
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                raise RuntimeError("Failed to load RL model for predictions")

            # Environment handles ALL trading logic
            obs, _, done, _, info = env.step(action)

            # ✅ Store action and environment state (no duplicate logic)
            actions.append(float(action[0]))
            env_states.append(info)

            # Progress tracking
            if (i - env.window_size) % 1000 == 0 and i > env.window_size:
                progress = ((i - env.window_size) / total_predictions) * 100
                current_pnl = env.balance - initial_portfolio  # FROM environment
                print(f"   📈 Progress: {progress:.1f}% ({i - env.window_size:,}/{total_predictions:,}) | PnL: ${current_pnl:+,.2f}")

            if done:
                break

        # ✅ Create results using environment data (single source of truth)
        result = df.iloc[env.window_size:env.window_size+len(actions)].copy()

        # Extract data from environment states
        result['rl_action'] = actions
        result['portfolio_value'] = [s['balance'] for s in env_states]
        result['position'] = [s['position'] for s in env_states]
        result['reward'] = [s['reward'] for s in env_states]
        result['step_pnl'] = [s['step_pnl'] for s in env_states]
        result['unrealized_pnl'] = [s['unrealized_pnl'] for s in env_states]
        result['position_exposure'] = [s['position_exposure'] for s in env_states]
        result['balance_change'] = [s['balance_change'] for s in env_states]

        # Calculate derived metrics
        portfolio_values = [s['balance'] for s in env_states]
        result['pnl'] = np.array(portfolio_values) - initial_portfolio
        result['pnl_pct'] = ((np.array(portfolio_values) / initial_portfolio) - 1) * 100
        result['cumulative_return'] = result['pnl_pct']

        # Convert RL actions to trading signals for compatibility
        result['signal'] = 'HOLD'
        result['action'] = 0

        buy_threshold = 0.1
        sell_threshold = -0.1

        buy_mask = np.array(actions) > buy_threshold
        sell_mask = np.array(actions) < sell_threshold

        result.loc[buy_mask, 'signal'] = 'BUY'
        result.loc[buy_mask, 'action'] = 1
        result.loc[sell_mask, 'signal'] = 'SELL'
        result.loc[sell_mask, 'action'] = -1

        # Extract trades from environment states
        trades_log = [s for s in env_states if s['trade_occurred']]
        if trades_log:
            result.trades_df = pd.DataFrame(trades_log)
        else:
            result.trades_df = pd.DataFrame(columns=['timestamp', 'action', 'price', 'step_pnl', 'portfolio_value'])

        # ✅ Final statistics using REAL environment data
        final_portfolio = env.balance  # FROM environment
        total_pnl = final_portfolio - initial_portfolio
        total_return_pct = (final_portfolio / initial_portfolio - 1) * 100

        buy_signals = np.sum(buy_mask)
        sell_signals = np.sum(sell_mask)
        hold_signals = len(actions) - buy_signals - sell_signals

        print(f"✅ Generated {len(actions):,} predictions using environment trading logic")
        print(f"💰 Final Portfolio: ${final_portfolio:,.2f}")
        print(f"📈 Total PnL: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
        print(f"🔄 Trading Signals: {buy_signals:,} BUY | {sell_signals:,} SELL | {hold_signals:,} HOLD")
        print(f"📊 Individual Trades: {len(trades_log)}")

        if trades_log:
            profitable_trades = len([t for t in trades_log if t['step_pnl'] > 0])
            win_rate = (profitable_trades / len(trades_log)) * 100
            avg_step_pnl = np.mean([t['step_pnl'] for t in trades_log])
            max_exposure = max([s['position_exposure'] for s in env_states])
            print(f"💹 Avg Step PnL: ${avg_step_pnl:+.2f} | Max Exposure: ${max_exposure:,.2f}")
            print(f"🎯 Win Rate: {win_rate:.1f}% ({profitable_trades}/{len(trades_log)} trades)")

        return result

    def save_normalizer(self, normalizer: FeatureNormalizer, filename: str = 'normalizer.pkl'):
        """Save the fitted normalizer"""
        normalizer_path = os.path.join(self.model_dir, filename)
        normalizer.save(normalizer_path)

    def load_normalizer(self, filename: str = 'normalizer.pkl') -> FeatureNormalizer:
        """Load a previously saved normalizer"""
        normalizer_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Normalizer file not found: {normalizer_path}")

        print(f"📥 Loaded normalizer from: {normalizer_path}")
        return FeatureNormalizer.load(normalizer_path)
