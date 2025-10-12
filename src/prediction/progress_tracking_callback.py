"""
Progress Tracking Callback for RL Training
Focused on training progress reporting and ETA calculations
"""

import time
from stable_baselines3.common.callbacks import BaseCallback


class ProgressTrackingCallback(BaseCallback):
    """
    Progress tracking callback for RL training
    
    Provides:
    - Real-time training progress reporting
    - Speed tracking (steps/second)
    - ETA calculations
    - Training session timing
    """

    def __init__(self, total_timesteps: int = 50000, verbose: int = 1):
        """
        Initialize the progress tracking callback

        Args:
            total_timesteps: Total training timesteps for progress calculation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None

        # Dynamic print interval based on total timesteps
        if total_timesteps <= 1000:
            self.print_interval = 250
        elif total_timesteps <= 5000:
            self.print_interval = 500
        else:
            self.print_interval = 1000

    def _on_training_start(self) -> None:
        """Called when training starts"""
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"🚀 Training started at {time.strftime('%H:%M:%S')}")
            print(f"📊 Target: {self.total_timesteps:,} timesteps")
            print("=" * 60)

    def _on_step(self) -> bool:
        """
        Called after each step - handles progress reporting

        Returns:
            bool: Always True to continue training
        """
        # Print progress at regular intervals
        if self.start_time and self.num_timesteps % self.print_interval == 0:
            self._print_progress()

        return True

    def _print_progress(self):
        """Print training progress with speed and ETA"""
        if not self.start_time:
            return

        current_time = time.time()
        elapsed = current_time - self.start_time
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
        progress_pct = (self.num_timesteps / self.total_timesteps) * 100

        # Calculate ETA
        if steps_per_sec > 0 and self.num_timesteps < self.total_timesteps:
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta_seconds = remaining_steps / steps_per_sec
            eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "00:00" if self.num_timesteps >= self.total_timesteps else "??:??"

        status = "✅ COMPLETED" if self.num_timesteps >= self.total_timesteps else "🔄 RUNNING"

        if self.verbose > 0:
            print(f"📈 Step {self.num_timesteps:,}/{self.total_timesteps:,} "
                  f"({progress_pct:.1f}%) | "
                  f"Speed: {steps_per_sec:.0f} steps/s | "
                  f"ETA: {eta_str} | {status}")

    def _on_training_end(self) -> None:
        """Called when training ends"""
        if self.start_time and self.verbose > 0:
            total_time = time.time() - self.start_time
            avg_speed = self.num_timesteps / total_time if total_time > 0 else 0
            print("=" * 60)
            print(f"✅ Training completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
            print(f"📊 Average speed: {avg_speed:.0f} steps/s")
            print(f"🎯 Final timesteps: {self.num_timesteps:,}")

    def get_training_stats(self) -> dict:
        """
        Get current training statistics

        Returns:
            dict: Training statistics including timing and progress
        """
        if not self.start_time:
            return {}

        elapsed = time.time() - self.start_time
        progress_pct = (self.num_timesteps / self.total_timesteps) * 100
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0

        return {
            'timesteps_completed': self.num_timesteps,
            'total_timesteps': self.total_timesteps,
            'progress_percent': progress_pct,
            'elapsed_time_seconds': elapsed,
            'steps_per_second': steps_per_sec,
            'start_time': self.start_time,
            'is_completed': self.num_timesteps >= self.total_timesteps
        }


def create_progress_tracking_callback(total_timesteps: int = 50000, verbose: int = 1) -> ProgressTrackingCallback:
    """
    Factory function to create progress tracking callbacks

    Args:
        total_timesteps: Total training timesteps for progress calculation
        verbose: Verbosity level

    Returns:
        ProgressTrackingCallback: Configured progress tracking callback
    """
    return ProgressTrackingCallback(
        total_timesteps=total_timesteps,
        verbose=verbose
    )