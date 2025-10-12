"""
Trading Metrics Callback for RL Training
Focused on capturing and reporting trading-specific metrics during evaluation
"""

import numpy as np
import json
import os
from stable_baselines3.common.callbacks import EvalCallback


class TradingMetricsCallback(EvalCallback):
    """
    Trading metrics callback that extends EvalCallback to capture trading-specific metrics
    
    Provides:
    - Enhanced trading metrics tracking from environment info
    - Portfolio performance monitoring
    - Trade analysis and win rate tracking
    - Comprehensive training data export
    """

    def __init__(self, eval_env, **kwargs):
        """
        Initialize the trading metrics callback

        Args:
            eval_env: Environment for evaluation
            **kwargs: Additional arguments passed to EvalCallback
        """
        super().__init__(eval_env, **kwargs)
        
        # Track trading metrics from enhanced environment info
        self.trading_metrics_history = []
        self.evaluation_rewards_history = []
        self.portfolio_performance_history = []

    def _on_step(self) -> bool:
        """
        Called after each step - handles evaluation and trading metrics capture

        Returns:
            bool: True to continue training, False to stop
        """
        # Call parent evaluation logic
        continue_training = super()._on_step()

        # Capture trading metrics if we just completed an evaluation
        if (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and
                hasattr(self, 'last_mean_reward') and hasattr(self, '_last_episode_infos')):

            # Store evaluation reward
            self.evaluation_rewards_history.append({
                'timestep': self.num_timesteps,
                'mean_reward': float(self.last_mean_reward),
                'std_reward': getattr(self, 'last_std_reward', 0.0)
            })

            # Capture enhanced trading metrics from episode infos
            if hasattr(self, '_last_episode_infos'):
                self._capture_trading_metrics_during_eval(
                    getattr(self, '_last_episode_rewards', []),
                    getattr(self, '_last_episode_lengths', []),
                    getattr(self, '_last_episode_infos', [])
                )

        return continue_training

    def _capture_trading_metrics_during_eval(self, episode_rewards, episode_lengths, episode_infos):
        """
        Capture enhanced trading metrics from evaluation episodes

        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            episode_infos: List of episode info dictionaries (contains trading data)
        """
        if not episode_infos:
            return

        # Aggregate trading metrics across evaluation episodes
        total_trades = 0
        total_pnl = 0.0
        total_exposure = 0.0
        profitable_trades = 0
        portfolio_values = []

        for episode_info_list in episode_infos:
            if not episode_info_list:
                continue

            # Get final episode info (last step of episode)
            final_info = episode_info_list[-1] if episode_info_list else {}

            # Extract trading metrics from enhanced environment info
            total_pnl += final_info.get('total_pnl', 0.0)
            portfolio_values.append(final_info.get('portfolio_value', 0.0))

            # Count trades and analyze performance across episode
            episode_trades = 0
            episode_profitable = 0
            max_exposure = 0.0

            for info in episode_info_list:
                if info.get('trade_occurred', False):
                    episode_trades += 1
                    if info.get('step_pnl', 0) > 0:
                        episode_profitable += 1

                max_exposure = max(max_exposure, info.get('position_exposure', 0.0))

            total_trades += episode_trades
            profitable_trades += episode_profitable
            total_exposure = max(total_exposure, max_exposure)

        # Calculate aggregated metrics
        avg_portfolio_value = np.mean(portfolio_values) if portfolio_values else 0.0
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_pnl = total_pnl / len(episode_infos) if episode_infos else 0.0

        # Store trading metrics for reporting
        trading_metrics = {
            'timestep': self.num_timesteps,
            'evaluation_step': len(self.trading_metrics_history),
            'avg_episode_reward': float(np.mean(episode_rewards)),
            'avg_portfolio_value': avg_portfolio_value,
            'avg_total_pnl': avg_pnl,
            'total_trades_per_episode': total_trades / len(episode_infos) if episode_infos else 0,
            'win_rate_pct': win_rate,
            'max_position_exposure': total_exposure,
            'episodes_evaluated': len(episode_infos)
        }

        self.trading_metrics_history.append(trading_metrics)

        # Store detailed portfolio performance
        portfolio_metrics = {
            'timestep': self.num_timesteps,
            'portfolio_values': portfolio_values,
            'total_pnl_values': [info_list[-1].get('total_pnl', 0) for info_list in episode_infos if info_list],
            'return_pct_values': [info_list[-1].get('total_return_pct', 0) for info_list in episode_infos if info_list]
        }

        self.portfolio_performance_history.append(portfolio_metrics)

        # Enhanced evaluation reporting
        if self.verbose > 0:
            print(f"📈 Eval #{len(self.trading_metrics_history)}: "
                  f"Reward: {np.mean(episode_rewards):.4f} | "
                  f"PnL: ${avg_pnl:+.2f} | "
                  f"Trades: {total_trades / len(episode_infos):.1f}/ep | "
                  f"Win Rate: {win_rate:.1f}%")

    def _on_training_end(self) -> None:
        """Called when training ends"""
        super()._on_training_end()
        
        # Save comprehensive training data including trading metrics
        self._save_enhanced_training_data()

    def _save_enhanced_training_data(self):
        """Save comprehensive training data including trading metrics"""
        # Prepare comprehensive training data
        training_data = {
            'session_info': {
                'algorithm': 'PPO',
                'actual_timesteps': self.num_timesteps,
                'eval_frequency': self.eval_freq,
                'n_eval_episodes': self.n_eval_episodes
            },
            'evaluation_rewards': self.evaluation_rewards_history,
            'trading_metrics': self.trading_metrics_history,
            'portfolio_performance': self.portfolio_performance_history,
            'final_metrics': {
                'training_completed': True,
                'final_timesteps': self.num_timesteps,
                'total_evaluations': len(self.trading_metrics_history)
            }
        }

        # Save to model directory
        if hasattr(self, 'best_model_save_path') and self.best_model_save_path:
            save_dir = os.path.dirname(self.best_model_save_path)
        else:
            save_dir = 'models/rl'

        os.makedirs(save_dir, exist_ok=True)

        # Save enhanced training data
        data_file = os.path.join(save_dir, 'trading_metrics_data.json')
        with open(data_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        if self.verbose > 0:
            print(f"💾 Trading metrics data saved to: {data_file}")

    def get_trading_summary(self) -> dict:
        """
        Get current trading performance summary

        Returns:
            dict: Trading performance metrics
        """
        if not self.trading_metrics_history:
            return {}

        latest_metrics = self.trading_metrics_history[-1]
        
        return {
            'total_evaluations': len(self.trading_metrics_history),
            'latest_win_rate': latest_metrics.get('win_rate_pct', 0),
            'latest_avg_pnl': latest_metrics.get('avg_total_pnl', 0),
            'latest_portfolio_value': latest_metrics.get('avg_portfolio_value', 0),
            'total_trades_tracked': sum([m.get('total_trades_per_episode', 0) for m in self.trading_metrics_history]),
            'evaluation_history_length': len(self.evaluation_rewards_history)
        }


def create_trading_metrics_callback(eval_env, **kwargs) -> TradingMetricsCallback:
    """
    Factory function to create trading metrics callbacks

    Args:
        eval_env: Environment for evaluation
        **kwargs: Additional arguments passed to TradingMetricsCallback

    Returns:
        TradingMetricsCallback: Configured trading metrics callback
    """
    return TradingMetricsCallback(eval_env, **kwargs)