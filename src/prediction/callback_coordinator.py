"""
Callback Coordinator for RL Training
Coordinates between early stopping and evaluation callbacks
"""

from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
from .early_stopping import EarlyStopping


class CallbackCoordinator(BaseCallback):
    """
    Coordinator callback that manages interaction between early stopping and evaluation callbacks
    """

    def __init__(self, early_stop_callback: Optional[EarlyStopping] = None, verbose: int = 1):
        super().__init__(verbose)
        self.early_stop_callback = early_stop_callback

    def _on_step(self) -> bool:
        """Check early stopping after each step if we have evaluation results"""
        # This gets called after evaluation callbacks have run
        if (self.early_stop_callback is not None and 
            hasattr(self, 'locals') and 
            self.locals and
            'last_mean_reward' in self.locals):
            
            mean_reward = self.locals['last_mean_reward']
            if isinstance(mean_reward, (int, float)):
                continue_training = self.early_stop_callback.check_early_stop(float(mean_reward))
                if not continue_training and self.verbose > 0:
                    print("🛑 Training stopped by early stopping callback")
                return continue_training
        
        return True


def create_callback_coordinator(early_stop_callback: Optional[EarlyStopping] = None, 
                              verbose: int = 1) -> CallbackCoordinator:
    """
    Factory function to create callback coordinator

    Args:
        early_stop_callback: Optional early stopping callback to coordinate with
        verbose: Verbosity level

    Returns:
        CallbackCoordinator: Configured callback coordinator
    """
    return CallbackCoordinator(
        early_stop_callback=early_stop_callback,
        verbose=verbose
    )