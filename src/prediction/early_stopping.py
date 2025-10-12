"""
Early Stopping Callback for RL Training
Focused purely on early stopping logic without dependencies
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingException(Exception):
    """Exception raised when early stopping is triggered"""
    pass


class EarlyStopping(BaseCallback):
    """
    Early stopping callback for RL training based on evaluation rewards
    Inherits from BaseCallback to work properly with SB3
    """

    def __init__(self, patience: int = 5, threshold: float = 0.01, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.threshold = threshold

        self.best_reward = -np.inf
        self.wait = 0
        self.best_timestep = 0

    def _on_step(self) -> bool:
        """Called after each step - we'll use this for early stopping logic"""
        return True

    def _on_training_start(self) -> None:
        """Called when training starts"""
        if self.verbose > 0:
            print(f"🎯 Early stopping initialized (patience={self.patience}, threshold={self.threshold})")

    def check_early_stop(self, mean_reward: float) -> bool:
        """
        Check if we should stop early based on evaluation reward

        Args:
            mean_reward: The mean reward from the latest evaluation

        Returns:
            bool: True to continue training, False to stop
        """
        current_timestep = self.num_timesteps

        # Check for improvement
        if mean_reward > self.best_reward + self.threshold:
            self.best_reward = mean_reward
            self.best_timestep = current_timestep
            self.wait = 0
            if self.verbose > 0:
                print(f"🎯 New best reward: {mean_reward:.4f} at timestep {current_timestep:,}")
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"⏳ No improvement for {self.wait}/{self.patience} evaluations "
                      f"(current: {mean_reward:.4f}, best: {self.best_reward:.4f})")

        # Check if we should stop
        if self.wait >= self.patience:
            if self.verbose > 0:
                print(f"🛑 Early stopping triggered! No improvement for {self.patience} evaluations")
                print(f"📊 Best reward: {self.best_reward:.4f} at timestep {self.best_timestep:,}")
            return False  # Stop training

        return True  # Continue training

    def reset(self):
        """Reset the early stopping state"""
        self.best_reward = -np.inf
        self.wait = 0
        self.best_timestep = 0
        if self.verbose > 0:
            print("🔄 Early stopping state reset")


def create_early_stopping_callback(patience: int = 5, threshold: float = 0.01, verbose: int = 1) -> EarlyStopping:
    """
    Factory function to create early stopping callbacks

    Args:
        patience: Number of evaluations to wait without improvement
        threshold: Minimum improvement threshold
        verbose: Verbosity level

    Returns:
        EarlyStopping: Configured early stopping callback
    """
    return EarlyStopping(
        patience=patience,
        threshold=threshold,
        verbose=verbose
    )
