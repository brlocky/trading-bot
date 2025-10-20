"""
Reinforcement Learning Predictor - Minimal RL trading agent using PPO
"""

from typing import Optional, Union
import pandas as pd
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure, Logger


import torch
from data_processing.feature_normalizer import FeatureNormalizer
from environments.trading_environment import TradingEnvironment
from core.normalization_config import (
    get_features_list,
    get_default_environment_config,
    get_model_config,
)


class RLPredictor:
    """
    Enhanced RL trading predictor using PPO with GPU support
    Uses standalone configuration and separated environment
    """

    def __init__(self, model_dir: str = 'models/rl'):
        self.model_dir = model_dir
        self.model: Optional[RecurrentPPO] = None
        self.normalizer: Optional[FeatureNormalizer] = None

        # Get default configurations
        self.feature_list = get_features_list()
        self.env_config = get_default_environment_config()
        self.train_config = get_model_config()

        # Detect available device
        self.device = self._detect_device()
        # self.device = self._detect_device()
        print(f"🖥️ RL Training Device: {self.device}")

        # Create vectorized training environment using helper method
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(self.model_dir, 'eval_logs')
        os.makedirs(self.log_dir, exist_ok=True)

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

    def get_model_config(self) -> dict:
        return {
            'window_size': self.train_config.get('window_size'),
            'initial_balance': self.env_config.get('initial_balance'),
            'buy_threshold': self.env_config.get('buy_threshold'),
            'sell_threshold': self.env_config.get('sell_threshold'),
        }

    def _create_vectorized_env(self, df: pd.DataFrame, normalizer: FeatureNormalizer, n_envs=1, use_log=False) -> DummyVecEnv:
        """
        Helper method to create a vectorized environment with configurable number of parallel environments

        Args:
            df: Training dataframe
            normalizer: Fitted feature normalizer
            log_dir: Directory for Monitor logs (optional)

        Returns:
            DummyVecEnv with configured number of parallel environments
        """

        def make_env(env_id: int):
            def _init():
                env = TradingEnvironment(
                    df=df,
                    normalizer=normalizer,
                    feature_config=self.feature_list,
                    **self.get_model_config()
                )
                if use_log:
                    monitor_filename = os.path.join(self.log_dir, f"monitor_env_{env_id}.csv")
                    return Monitor(env, filename=monitor_filename)
                return env
            return _init

        env_fns = []
        for i in range(n_envs):
            env_fns.append(make_env(i))

        return DummyVecEnv(env_fns)

    def get_model_and_normalizer(self,
                                 df: pd.DataFrame,
                                 continue_training: bool = False,
                                 n_envs: int = 1
                                 ) -> tuple[RecurrentPPO, FeatureNormalizer]:
        # Load existing model if continue_training=True, else create new
        if continue_training:
            print("🔄 Loading existing model...")
            normalizer = self.load_normalizer()
            ppo_model = self.load_model(None)
            print(f"✅ Loaded model with {ppo_model.num_timesteps:,} timesteps")
            return ppo_model, normalizer

        print("🆕 Creating new model...")
        # Configure PPO with settings from configuration
        policy_kwargs = {
            "net_arch": dict(
                pi=self.train_config['hidden_layers_pi'],    # [256, 256]
                vf=self.train_config['hidden_layers_vf']     # [256, 256]
            ),
            "lstm_hidden_size": self.train_config['lstm_hidden_size'],  # 256
            "activation_fn": getattr(torch.nn, self.train_config['activation_function'].title()),  # Remove ()
            "ortho_init": self.train_config['ortho_init']
        }

        normalizer = FeatureNormalizer(self.feature_list)
        df = df[self.feature_list.keys()].ffill().bfill().dropna()

        normalizer.fit(df, close_prices=df['close'])
        # Save the normalizer
        self.save_normalizer(normalizer)

        # Create dummy vectorized environment for model initialization
        dummy_vec_env = self._create_vectorized_env(df, normalizer, n_envs)

        batch_size = (self.train_config['batch_size_gpu'] if self.device == 'cuda'
                      else self.train_config.get('batch_size_cpu', self.train_config['batch_size_gpu']))
        dummy_vec_env.close()

        ppo_model = RecurrentPPO(
            policy='MlpLstmPolicy',        # LSTM policy
            env=dummy_vec_env,
            device=self.device,            # Add device parameter
            learning_rate=self.train_config['learning_rate'],
            n_steps=self.train_config['n_steps'],
            batch_size=batch_size,
            n_epochs=self.train_config['n_epochs'],
            gamma=self.train_config['gamma'],
            gae_lambda=self.train_config['gae_lambda'],
            clip_range=self.train_config['clip_range'],

            # NEW: Additional parameters for better training control
            clip_range_vf=self.train_config.get('clip_range_vf', None),
            normalize_advantage=self.train_config.get('normalize_advantage', True),
            target_kl=self.train_config.get('target_kl', None),
            stats_window_size=self.train_config.get('stats_window_size', 100),
            tensorboard_log=os.path.join(self.model_dir, 'tensorboard_logs'),
            seed=self.train_config.get('seed', None),

            ent_coef=self.train_config['ent_coef'],
            vf_coef=self.train_config['vf_coef'],
            max_grad_norm=self.train_config['max_grad_norm'],
            use_sde=self.train_config['use_sde'],
            sde_sample_freq=self.train_config['sde_sample_freq'],
            policy_kwargs=policy_kwargs,
            verbose=1,

        )

        return ppo_model, normalizer

    def train(self, train_df: pd.DataFrame, continue_training: bool = False, verbose: int = 0, **kwargs):
        """
        Train with proper data split and normalization to prevent data leakage
        Uses standalone configuration for all parameters, but allows overrides

        Args:
            df: Training data with features
            generate_report: Whether to generate a comprehensive training report after training
            verbose: Verbosity level for training logs
            continue_training: Whether to continue training from an existing model and normalizer if found
            callbacks: List of callbacks (including Keras-style callbacks)
            **kwargs: Override training configuration parameters
        """
        n_envs = self.train_config.get('n_envs', 4)

        print(f"🚀 Initializing PPO model on {self.device}...")
        self.model, self.normalizer = self.get_model_and_normalizer(df=train_df, continue_training=continue_training, n_envs=n_envs)

        if not self.model or not self.normalizer:
            raise RuntimeError("Failed to initialize model or normalizer")

        train_env = self._create_vectorized_env(train_df, self.normalizer, n_envs, True)

        print(f"📊 Training with {n_envs} parallel environments")
        print(f"📊 Monitor logs will be saved to: {self.log_dir}")

        # Assign the environment to the model
        self.model.set_env(train_env)

        # ✅ CONFIGURE COMPREHENSIVE LOGGING: Set up SB3's built-in logging
        # Configure logger with multiple output formats
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

        try:
            total_timesteps = kwargs.get('total_timesteps', self.train_config['total_timesteps'])
            log_interval = kwargs.get('log_interval', 1)

            print(f"🚀 Starting PPO training for {total_timesteps:,} timesteps on {self.device}...")

            self.model.learn(
                total_timesteps=total_timesteps,
                log_interval=log_interval,
                progress_bar=True,
            )
            print("✅ Training completed successfully")

            # ✅ FIX: Clean shutdown environment before saving
            train_env.close()

            # Save model only on successful completion
            model_path = os.path.join(self.model_dir, 'ppo_trading')
            self.model.save(model_path)
            print(f"✅ Model saved to: {model_path}.zip")

            # ✅ FIX: Only assign to self.model after successful save
            training_successful = True

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user - model NOT saved")
            training_successful = False
        except Exception as e:
            print(f"❌ Training failed: {e} - model NOT saved")
            training_successful = False
            raise
        finally:
            # ✅ FIX: Ensure cleanup even if training fails
            try:
                train_env.close()
            except Exception:
                pass

        return training_successful  # Indicate successful training

    def load_model(self, env: Optional[TradingEnvironment]) -> RecurrentPPO:
        try:
            """Load the trained model with proper device handling"""
            path = os.path.join(self.model_dir, 'ppo_trading.zip')
            # Load model with device specification
            if env:
                model = RecurrentPPO.load(path, device=self.device, env=env)
            else:
                model = RecurrentPPO.load(path, device=self.device)
            print(f"✅ Model loaded on {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def save_normalizer(self, normalizer: FeatureNormalizer, filename: str = 'normalizer.pkl'):
        """Save the fitted normalizer"""
        if not normalizer.is_fitted:
            raise ValueError("Cannot save an unfitted normalizer")
        normalizer_path = os.path.join(self.model_dir, filename)
        normalizer.save(normalizer_path)

    def load_normalizer(self, filename: str = 'normalizer.pkl') -> FeatureNormalizer:
        """Load a previously saved normalizer"""
        try:
            normalizer_path = os.path.join(self.model_dir, filename)
            print(f"📥 Loaded normalizer from: {normalizer_path}")
            return FeatureNormalizer.load(normalizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load normalizer: {e}")

    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading predictions (signals and position sizes) from the trained RL model

        Args:
            df: DataFrame with OHLCV data and features

        Returns:
            DataFrame with columns: signal, position_size, confidence
        """
        print(f"🎯 Generating predictions for {len(df)} candles...")

        self.normalizer = self.load_normalizer()

        # Create environment for predictions
        env = TradingEnvironment(df, **self.get_model_config(), feature_config=self.feature_list, normalizer=self.normalizer)

        # Load model with environment
        self.model = self.load_model(env)  # need to load the env on the load because it diff from the 8 on the training

        # Generate predictions
        obs, _ = env.reset()

        last_state = None
        while True:
            try:
                # Get model action - wrap in try/catch to catch internal NaN errors
                action, _states = self.model.predict(obs, last_state, deterministic=True)
                last_state = _states
            except ValueError as e:
                raise e

            # Handle Box action space: [signal, position_size]
            # Extract signal and position size from Box action
            raw_signal = float(action[0])  # Range: -1.0 to 1.0
            raw_position_size = float(action[1])  # Range: 0.0 to 1.0

            # Convert continuous signal to discrete: -1, 0, 1
            # Use more sensitive thresholds to allow trading
            if raw_signal > self.env_config['buy_threshold']:  # Lower threshold for BUY
                signal = 1  # BUY
            elif raw_signal < self.env_config['sell_threshold']:  # Lower threshold for SELL
                signal = -1  # SELL
            else:
                signal = 0  # HOLD

            # Step environment to get next observation
            obs, reward, done, truncated, info = env.step([signal, raw_position_size])

            if done or truncated:
                break

        # Create results DataFrame
        if len(env.broker.step_history) == 0:
            print("⚠️ No predictions generated - creating empty DataFrame with correct structure")
            # Create empty DataFrame
            return pd.DataFrame()

        results_df = pd.DataFrame(env.broker.step_history)
        results_df.index = df.index[self.train_config['window_size']:self.train_config['window_size'] + len(env.broker.step_history)]

        return results_df
