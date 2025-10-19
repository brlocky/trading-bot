"""
Model Training Report Generator with Enhanced Broker Integration

RL training report generator that uses simplified broker step history for comprehensive
trading performance analysis with direct line chart visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class ModelTrainingReport:
    """Enhanced RL model training report generator with simplified broker integration."""

    def __init__(self, model_dir: str = 'models/rl_demo'):
        """Initialize the training report generator."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_info = None
        self.training_data = None

    def show_training_report(self, show_charts: bool = True) -> Dict:
        """
        Generate and display comprehensive PPO training performance report

        Args:
            show_charts: Whether to display interactive charts (set False for headless environments)

        Returns:
            Dict containing all training metrics and analysis
        """
        print("🚀 PPO Training Performance Report")
        print("=" * 60)

        # Load training data
        self._load_training_data()
        if not self.training_data:
            print("❌ No training data found. Make sure training was completed with TrainingReportCallback.")
            return {}

        # Generate analysis
        analysis = self._analyze_training_performance()

        # Display text report
        self._display_text_report(analysis)

        # Display charts if requested
        if show_charts:
            self._display_training_charts(analysis)

        return analysis

    def _load_training_data(self) -> None:
        """Load training data from JSON file"""
        try:
            # Fallback to standard training data
            data_file = self.model_dir / 'training_logs.json'
            if data_file.exists():
                with open(data_file, 'r') as f:
                    self.training_data = json.load(f)
                print("📊 Loaded standard training data")
                return

        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            self.training_data = None

    def _analyze_training_performance(self) -> Dict:
        """Analyze training performance using simplified broker step history"""
        if not self.training_data:
            return {}

        session_info = self.training_data.get('session_info', {})
        step_metrics = self.training_data.get('step_metrics', [])
        final_metrics = self.training_data.get('final_metrics', {})

        # ✅ NEW: Use simplified broker step history
        broker_history = self.training_data.get('broker_step_history', [])
        evaluation_rewards = self.training_data.get('evaluation_rewards', [])

        analysis = {
            'session_overview': {
                'algorithm': session_info.get('algorithm', 'PPO'),
                'total_timesteps': session_info.get('total_timesteps', 0),
                'actual_timesteps': final_metrics.get('final_timesteps', 0),
                'duration_seconds': session_info.get('training_duration', 0),
                'avg_speed': final_metrics.get('average_steps_per_second', 0),
                'start_time': session_info.get('start_time'),
                'end_time': session_info.get('end_time'),
                'completed': final_metrics.get('training_completed', False),
                'total_evaluations': final_metrics.get('total_evaluations', 0)
            },
            'performance_metrics': self._analyze_ppo_metrics(step_metrics, final_metrics),
            'training_stability': self._assess_training_stability(step_metrics),

            # ✅ NEW: Simplified broker-based analysis
            'broker_performance': self._analyze_broker_step_history(broker_history),
            'evaluation_progress': self._analyze_evaluation_rewards(evaluation_rewards),
            'trading_charts_data': self._prepare_trading_charts_data(broker_history),

            'recommendations': self._generate_recommendations(step_metrics, final_metrics, broker_history)
        }

        return analysis

    def _analyze_ppo_metrics(self, step_metrics: List[Dict], final_metrics: Dict) -> Dict:
        """Analyze PPO-specific metrics"""
        if not step_metrics:
            return {}

        # Extract final values
        final_perf = final_metrics.get('performance', {})

        metrics = {
            'policy_loss': {
                'final': final_perf.get('policy_gradient_loss'),
                'description': 'Policy gradient loss - lower is better'
            },
            'value_loss': {
                'final': final_perf.get('value_loss'),
                'description': 'Value function loss - measures critic learning'
            },
            'entropy': {
                'final': final_perf.get('entropy_loss'),
                'description': 'Entropy loss - measures exploration vs exploitation'
            },
            'clip_fraction': {
                'final': final_perf.get('clip_fraction'),
                'description': 'Fraction of policy updates that were clipped'
            },
            'explained_variance': {
                'final': final_perf.get('explained_variance'),
                'description': 'How well value function explains returns'
            },
            'learning_rate': {
                'final': final_perf.get('learning_rate'),
                'description': 'Current learning rate'
            }
        }

        return metrics

    def _analyze_loss_trends(self, loss_history: List[Dict], step_metrics: List[Dict]) -> Dict:
        """Analyze loss trends over training"""
        trends = {}

        if loss_history:
            # Group by loss type
            loss_types = {}
            for entry in loss_history:
                loss_type = entry['loss_type']
                if loss_type not in loss_types:
                    loss_types[loss_type] = []
                loss_types[loss_type].append({
                    'timestep': entry['timestep'],
                    'value': entry['value']
                })
            trends['detailed'] = loss_types

        # Extract from step metrics if loss_history is empty
        if not loss_history and step_metrics:
            policy_losses = []
            value_losses = []
            entropy_losses = []

            for step in step_metrics:
                if 'policy_gradient_loss' in step:
                    policy_losses.append({
                        'timestep': step['timestep'],
                        'value': step['policy_gradient_loss']
                    })
                if 'value_loss' in step:
                    value_losses.append({
                        'timestep': step['timestep'],
                        'value': step['value_loss']
                    })
                if 'entropy_loss' in step:
                    entropy_losses.append({
                        'timestep': step['timestep'],
                        'value': step['entropy_loss']
                    })

            trends['from_steps'] = {
                'policy_gradient_loss': policy_losses,
                'value_loss': value_losses,
                'entropy_loss': entropy_losses
            }

        return trends

    def _assess_training_stability(self, step_metrics: List[Dict]) -> Dict:
        """Assess training stability and convergence"""
        if not step_metrics or len(step_metrics) < 2:
            return {'status': 'insufficient_data'}

        # Check speed stability
        speeds = [step.get('elapsed_time', 0) for step in step_metrics if step.get('elapsed_time')]
        if speeds:
            # Calculate steps per second for each interval
            step_speeds = []
            for i in range(1, len(step_metrics)):
                time_diff = step_metrics[i]['elapsed_time'] - step_metrics[i-1]['elapsed_time']
                step_diff = step_metrics[i]['timestep'] - step_metrics[i-1]['timestep']
                if time_diff > 0:
                    step_speeds.append(step_diff / time_diff)

            if step_speeds:
                avg_speed = sum(step_speeds) / len(step_speeds)
                speed_variance = sum((s - avg_speed) ** 2 for s in step_speeds) / len(step_speeds)
                speed_stability = 'stable' if speed_variance < (avg_speed * 0.1) ** 2 else 'variable'
            else:
                speed_stability = 'unknown'
        else:
            speed_stability = 'unknown'

        return {
            'speed_stability': speed_stability,
            'data_points': len(step_metrics),
            'training_progression': 'completed' if step_metrics else 'incomplete'
        }

    def _convert_old_format(self, old_data: Dict) -> Dict:
        """Convert old training_logs.json format to new format"""
        return {
            'session_info': old_data.get('session_info', {}),
            'step_metrics': old_data.get('performance_metrics', []),
            'loss_history': [],
            'final_metrics': old_data.get('final_metrics', {})
        }

    def _generate_recommendations(
        self,
        step_metrics: List[Dict],
        final_metrics: Dict,
        trading_metrics: Optional[List[Dict]] = None
    ) -> List[str]:
        """Generate training improvement recommendations"""
        recommendations = []
        final_perf = final_metrics.get('performance', {})

        # Analyze clip fraction
        clip_fraction = final_perf.get('clip_fraction', 0)
        if clip_fraction > 0.3:
            recommendations.append("⚠️ High clip fraction (>0.3) - consider reducing learning rate")
        elif clip_fraction < 0.05:
            recommendations.append("📈 Low clip fraction (<0.05) - could increase learning rate for faster learning")
        else:
            recommendations.append("✅ Clip fraction in optimal range (0.05-0.3)")

        # Analyze explained variance
        explained_var = final_perf.get('explained_variance', 0)
        if explained_var < 0:
            recommendations.append("⚠️ Negative explained variance - value function struggling to learn")
        elif explained_var > 0.5:
            recommendations.append("✅ Good explained variance - value function learning well")
        else:
            recommendations.append("📊 Moderate explained variance - value function making progress")

        # Analyze entropy
        entropy = final_perf.get('entropy_loss', 0)
        if entropy > -1.0:
            recommendations.append("🎲 High entropy - model still exploring (good for early training)")
        elif entropy < -3.0:
            recommendations.append("🎯 Low entropy - model focused on exploitation (good for late training)")
        else:
            recommendations.append("⚖️ Balanced exploration/exploitation")

        # ✅ NEW: Trading performance recommendations
        if trading_metrics:
            self._add_trading_recommendations(recommendations, trading_metrics)

        # Training duration analysis
        duration = final_metrics.get('total_duration_seconds', 0)
        timesteps = final_metrics.get('final_timesteps', 0)
        if duration > 0 and timesteps > 0:
            steps_per_sec = timesteps / duration
            if steps_per_sec < 100:
                recommendations.append("⏱️ Slow training speed - consider GPU acceleration or simpler environment")
            elif steps_per_sec > 1000:
                recommendations.append("🚀 Excellent training speed - setup is well optimized")

        return recommendations

    def _display_text_report(self, analysis: Dict) -> None:
        """Display comprehensive text report"""
        session = analysis.get('session_overview', {})
        metrics = analysis.get('performance_metrics', {})
        stability = analysis.get('training_stability', {})
        recommendations = analysis.get('recommendations', [])

        # ✅ NEW: Extract trading performance data
        trading_perf = analysis.get('trading_performance', {})
        eval_progress = analysis.get('evaluation_progress', {})
        portfolio_analysis = analysis.get('portfolio_analysis', {})

        print("📊 SESSION OVERVIEW")
        print("=" * 30)
        print(f"Algorithm: {session.get('algorithm', 'Unknown')}")
        print(f"Timesteps: {session.get('actual_timesteps', 0):,} / {session.get('total_timesteps', 0):,}")
        print(f"Evaluations: {session.get('total_evaluations', 0)}")

        duration = session.get('duration_seconds', 0)
        if duration > 0:
            print(f"Duration: {int(duration//60):02d}:{int(duration%60):02d}")
            print(f"Speed: {session.get('avg_speed', 0):.0f} steps/sec")

        print(f"Status: {'✅ Completed' if session.get('completed') else '❌ Incomplete'}")

        print("\n🎯 PPO PERFORMANCE METRICS")
        print("=" * 35)

        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'final' in metric_data:
                value = metric_data['final']
                desc = metric_data.get('description', '')
                if value is not None:
                    if 'loss' in metric_name:
                        print(f"{metric_name.replace('_', ' ').title()}: {value:.6f}")
                    else:
                        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
                    if desc:
                        print(f"  └─ {desc}")
                else:
                    print(f"{metric_name.replace('_', ' ').title()}: Not available")

        # ✅ NEW: Display trading performance metrics
        if trading_perf.get('status') != 'no_data':
            print("\n� TRADING PERFORMANCE")
            print("=" * 25)
            print(f"Avg Trades per Episode: {trading_perf.get('avg_trades_per_episode', 0):.2f}")
            print(f"Average Win Rate: {trading_perf.get('avg_win_rate_pct', 0):.1f}%")
            print(f"Avg PnL per Episode: ${trading_perf.get('avg_pnl_per_episode', 0):+.2f}")
            print(f"Max Position Exposure: ${trading_perf.get('max_position_exposure', 0):,.2f}")
            print(f"Final Portfolio Value: ${trading_perf.get('final_portfolio_value', 0):,.2f}")
            print(f"Trading Activity: {trading_perf.get('trading_activity', 'Unknown')}")

        # ✅ NEW: Display evaluation progress
        if eval_progress.get('status') != 'no_data':
            print("\n📈 EVALUATION PROGRESS")
            print("=" * 25)
            print(f"Initial Reward: {eval_progress.get('initial_reward', 0):.4f}")
            print(f"Final Reward: {eval_progress.get('final_reward', 0):.4f}")
            print(f"Best Reward: {eval_progress.get('best_reward', 0):.4f}")
            print(f"Improvement: {eval_progress.get('improvement', 0):+.4f}")

            if 'pnl_trend' in eval_progress:
                pnl_trend = eval_progress['pnl_trend']
                print(f"PnL Progress: ${pnl_trend.get('initial_pnl', 0):+.2f} → ${pnl_trend.get('final_pnl', 0):+.2f}")

            if 'win_rate_trend' in eval_progress:
                wr_trend = eval_progress['win_rate_trend']
                print(f"Win Rate Progress: {wr_trend.get('initial_win_rate', 0):.1f}% → {wr_trend.get('final_win_rate', 0):.1f}%")

        # ✅ NEW: Display portfolio analysis
        if portfolio_analysis.get('status') != 'no_data':
            print("\n📊 PORTFOLIO ANALYSIS")
            print("=" * 25)
            print(f"Avg Portfolio Value: ${portfolio_analysis.get('avg_portfolio_value', 0):,.2f}")
            print(
                f"Portfolio Range: ${portfolio_analysis.get('min_portfolio_value', 0):,.2f} - "
                f"${portfolio_analysis.get('max_portfolio_value', 0):,.2f}")
            print(f"Portfolio Volatility: ${portfolio_analysis.get('portfolio_volatility', 0):,.2f}")
            print(f"Avg Return: {portfolio_analysis.get('avg_return_pct', 0):+.2f}%")
            print(
                f"Return Range: {portfolio_analysis.get('min_return_pct', 0):+.2f}% - "
                f"{portfolio_analysis.get('max_return_pct', 0):+.2f}%")

        print("\n�🔍 TRAINING STABILITY")
        print("=" * 25)
        print(f"Speed Stability: {stability.get('speed_stability', 'Unknown')}")
        print(f"Data Points: {stability.get('data_points', 0)}")
        print(f"Progression: {stability.get('training_progression', 'Unknown')}")

        print("\n💡 RECOMMENDATIONS")
        print("=" * 20)
        for rec in recommendations:
            print(f"  {rec}")

        print("\n📋 COPY-FRIENDLY SUMMARY FOR AI ANALYSIS")
        print("=" * 45)
        print("Training Configuration:")
        print("- Algorithm: PPO")
        print(f"- Timesteps: {session.get('actual_timesteps', 0):,}")
        print(f"- Duration: {duration:.1f}s")
        print(f"- Speed: {session.get('avg_speed', 0):.0f} steps/sec")
        print(f"- Evaluations: {session.get('total_evaluations', 0)}")

        print("\nFinal Metrics:")
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'final' in metric_data and metric_data['final'] is not None:
                value = metric_data['final']
                print(f"- {metric_name}: {value:.6f}")

        # ✅ NEW: Add trading metrics to summary
        if trading_perf.get('status') != 'no_data':
            print("\nTrading Performance:")
            print(f"- Win Rate: {trading_perf.get('avg_win_rate_pct', 0):.1f}%")
            print(f"- Avg Trades/Episode: {trading_perf.get('avg_trades_per_episode', 0):.2f}")
            print(f"- Avg PnL/Episode: ${trading_perf.get('avg_pnl_per_episode', 0):+.2f}")

        print("\nKey Issues/Strengths:")
        for rec in recommendations:
            # Clean emojis for AI analysis
            clean_rec = (rec.replace('⚠️', 'WARNING:').replace('✅', 'GOOD:')
                         .replace('📈', 'NOTE:').replace('🎲', 'INFO:')
                         .replace('🎯', 'INFO:').replace('⚖️', 'BALANCED:')
                         .replace('⏱️', 'PERFORMANCE:').replace('🚀', 'EXCELLENT:')
                         .replace('📊', 'MODERATE:').replace('💰', 'TRADING:')
                         .replace('📉', 'DECLINING:').replace('❌', 'CRITICAL:'))
            print(f"- {clean_rec}")

    def _display_training_charts(self, analysis: Dict) -> None:
        """Display interactive training charts using plotly"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio

            # Configure plotly for notebook
            pio.renderers.default = "notebook"

        except ImportError:
            print("⚠️ Plotly not available - skipping charts. Install with: pip install plotly")
            return

        if not self.training_data:
            print("❌ No training data available for charts")
            return

        step_metrics = self.training_data.get('step_metrics', [])

        if not step_metrics:
            print("❌ No step metrics available for charts")
            return

        print("\n📈 INTERACTIVE TRAINING CHARTS")
        print("=" * 35)

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Policy Loss Over Time', 'Value Loss Over Time',
                'Entropy Loss Over Time', 'Clip Fraction Over Time',
                'Learning Rate Over Time', 'Explained Variance Over Time'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        # Extract data for plotting
        timesteps = [step['timestep'] for step in step_metrics]

        # Plot Policy Loss
        policy_losses = [step.get('policy_gradient_loss') for step in step_metrics]
        if any(x is not None for x in policy_losses):
            fig.add_trace(
                go.Scatter(x=timesteps, y=policy_losses, name='Policy Loss',
                           line=dict(color='red', width=2)),
                row=1, col=1
            )

        # Plot Value Loss
        value_losses = [step.get('value_loss') for step in step_metrics]
        if any(x is not None for x in value_losses):
            fig.add_trace(
                go.Scatter(x=timesteps, y=value_losses, name='Value Loss',
                           line=dict(color='blue', width=2)),
                row=1, col=2
            )

        # Plot Entropy Loss
        entropy_losses = [step.get('entropy_loss') for step in step_metrics]
        if any(x is not None for x in entropy_losses):
            fig.add_trace(
                go.Scatter(x=timesteps, y=entropy_losses, name='Entropy Loss',
                           line=dict(color='green', width=2)),
                row=2, col=1
            )

        # Plot Clip Fraction
        clip_fractions = [step.get('clip_fraction') for step in step_metrics]
        if any(x is not None for x in clip_fractions):
            fig.add_trace(
                go.Scatter(x=timesteps, y=clip_fractions, name='Clip Fraction',
                           line=dict(color='orange', width=2)),
                row=2, col=2
            )

        # Plot Learning Rate
        learning_rates = [step.get('learning_rate') for step in step_metrics]
        if any(x is not None for x in learning_rates):
            fig.add_trace(
                go.Scatter(x=timesteps, y=learning_rates, name='Learning Rate',
                           line=dict(color='purple', width=2)),
                row=3, col=1
            )

        # Plot Explained Variance
        explained_vars = [step.get('explained_variance') for step in step_metrics]
        if any(x is not None for x in explained_vars):
            fig.add_trace(
                go.Scatter(x=timesteps, y=explained_vars, name='Explained Variance',
                           line=dict(color='brown', width=2)),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="PPO Training Metrics Dashboard",
            title_x=0.5,
            showlegend=False,
            template="plotly_white"
        )

        # Update x-axes
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Timesteps", row=i, col=j)

        # Show the plot
        fig.show()

        print("✅ Training charts displayed above")
        print("💡 Tip: Hover over lines for detailed values, zoom and pan to explore")

    # Legacy methods for backward compatibility
    def generate_complete_report(self) -> Dict:
        """Legacy method - use show_training_report() instead"""
        print("⚠️ generate_complete_report() is deprecated, use show_training_report() instead")
        return self.show_training_report()

    def create_session_summary(self) -> Dict:
        """Legacy method - use show_training_report() instead"""
        print("⚠️ create_session_summary() is deprecated, use show_training_report() instead")
        return self.show_training_report()

    # ✅ NEW: Simplified broker-based analysis methods
    def _analyze_broker_step_history(self, broker_history: List[Dict]) -> Dict:
        """Analyze broker performance from step history"""
        if not broker_history:
            return {'status': 'no_data'}

        # Extract key metrics from step history
        total_steps = len(broker_history)
        trade_steps = [step for step in broker_history if step.get('trade_occurred')]

        # Calculate trading statistics
        total_trades = len(trade_steps)
        trade_frequency = total_trades / total_steps if total_steps > 0 else 0

        # Portfolio progression
        portfolio_values = [step.get('portfolio_value', 0) for step in broker_history]
        initial_portfolio = portfolio_values[0] if portfolio_values else 0
        final_portfolio = portfolio_values[-1] if portfolio_values else 0

        # Position analysis
        position_sizes = [abs(step.get('position_shares', 0)) for step in broker_history]
        max_position = max(position_sizes) if position_sizes else 0

        # P&L analysis
        step_pnls = [step.get('step_pnl', 0) for step in broker_history]
        positive_pnl_steps = len([pnl for pnl in step_pnls if pnl > 0])
        win_rate = positive_pnl_steps / len(step_pnls) if step_pnls else 0

        return {
            'total_steps': total_steps,
            'total_trades': total_trades,
            'trade_frequency': trade_frequency,
            'initial_portfolio': initial_portfolio,
            'final_portfolio': final_portfolio,
            'portfolio_return': ((final_portfolio - initial_portfolio) / initial_portfolio * 100) if initial_portfolio > 0 else 0,
            'max_position_size': max_position,
            'win_rate_pct': win_rate * 100,
            'avg_step_pnl': sum(step_pnls) / len(step_pnls) if step_pnls else 0,
            'profitable_steps': positive_pnl_steps,
            'status': 'analyzed'
        }

    def _analyze_evaluation_rewards(self, evaluation_rewards: List[Dict]) -> Dict:
        """Analyze evaluation reward progression"""
        if not evaluation_rewards:
            return {'status': 'no_data'}

        rewards = [r.get('mean_reward', 0) for r in evaluation_rewards]

        return {
            'initial_reward': rewards[0] if rewards else 0,
            'final_reward': rewards[-1] if rewards else 0,
            'best_reward': max(rewards) if rewards else 0,
            'worst_reward': min(rewards) if rewards else 0,
            'total_evaluations': len(rewards),
            'improvement': (rewards[-1] - rewards[0]) if len(rewards) > 1 else 0,
            'reward_trend': 'improving' if len(rewards) > 1 and rewards[-1] > rewards[0] else 'declining' if len(rewards) > 1 else 'stable',
            'status': 'analyzed'
        }

    def _prepare_trading_charts_data(self, broker_history: List[Dict]) -> Dict:
        """Prepare data for direct line chart visualization"""
        if not broker_history:
            return {'status': 'no_data'}

        # Extract time series data for charts
        steps = [step.get('step', i) for i, step in enumerate(broker_history)]
        prices = [step.get('price', 0) for step in broker_history]
        portfolio_values = [step.get('portfolio_value', 0) for step in broker_history]
        position_shares = [step.get('position_shares', 0) for step in broker_history]
        balances = [step.get('balance', 0) for step in broker_history]
        step_pnls = [step.get('step_pnl', 0) for step in broker_history]
        unrealized_pnls = [step.get('unrealized_pnl', 0) for step in broker_history]

        # Trade markers (for scatter overlay)
        trade_steps = [step.get('step', i) for i, step in enumerate(broker_history) if step.get('trade_occurred')]
        trade_prices = [step.get('price', 0) for step in broker_history if step.get('trade_occurred')]
        trade_signals = [step.get('signal', 0) for step in broker_history if step.get('trade_occurred')]

        return {
            'time_series': {
                'steps': steps,
                'prices': prices,
                'portfolio_values': portfolio_values,
                'position_shares': position_shares,
                'balances': balances,
                'step_pnls': step_pnls,
                'unrealized_pnls': unrealized_pnls,
            },
            'trade_markers': {
                'steps': trade_steps,
                'prices': trade_prices,
                'signals': trade_signals
            },
            'total_data_points': len(broker_history),
            'status': 'ready_for_charts'
        }

    def _analyze_trading_performance(self, broker_history: List[Dict]) -> Dict:
        """Analyze trading performance from broker step history"""
        if not broker_history:
            return {'status': 'no_trading_data'}

        # Use the broker step history analysis
        return self._analyze_broker_step_history(broker_history)

    def _analyze_evaluation_progress(self, evaluation_rewards: List[Dict], broker_history: List[Dict]) -> Dict:
        """Analyze progression of evaluation rewards and trading performance"""
        if not evaluation_rewards:
            return {'status': 'no_data'}

        # Use the specific evaluation rewards analysis
        rewards_analysis = self._analyze_evaluation_rewards(evaluation_rewards)

        # Add broker trading trends if available
        if broker_history:
            broker_analysis = self._analyze_broker_step_history(broker_history)
            rewards_analysis['broker_trends'] = {
                'portfolio_return': broker_analysis.get('portfolio_return', 0),
                'win_rate': broker_analysis.get('win_rate_pct', 0),
                'trade_frequency': broker_analysis.get('trade_frequency', 0)
            }

        return rewards_analysis

    def _analyze_portfolio_performance(self, broker_history: List[Dict]) -> Dict:
        """Analyze portfolio performance from broker step history"""
        if not broker_history:
            return {'status': 'no_data'}

        # Extract portfolio values from broker history
        portfolio_values = [step.get('portfolio_value', 0) for step in broker_history]
        step_pnls = [step.get('step_pnl', 0) for step in broker_history]

        if not portfolio_values:
            return {'status': 'no_portfolio_data'}

        # Calculate statistics using numpy
        import numpy as np

        return {
            'status': 'analyzed',
            'avg_portfolio_value': float(np.mean(portfolio_values)),
            'max_portfolio_value': float(np.max(portfolio_values)),
            'min_portfolio_value': float(np.min(portfolio_values)),
            'portfolio_volatility': float(np.std(portfolio_values)),
            'avg_step_pnl': float(np.mean(step_pnls)) if step_pnls else 0,
            'total_portfolio_return': (
                (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
            ) if portfolio_values[0] > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'steps_analyzed': len(broker_history)
        }

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _add_trading_recommendations(self, recommendations: List[str], broker_history: List[Dict]) -> None:
        """Add trading-specific recommendations based on broker history"""
        if not broker_history:
            return

        broker_analysis = self._analyze_broker_step_history(broker_history)
        win_rate = broker_analysis.get('win_rate_pct', 0)
        trade_frequency = broker_analysis.get('trade_frequency', 0)
        portfolio_return = broker_analysis.get('portfolio_return', 0)

        # Win rate analysis
        if win_rate > 60:
            recommendations.append("🎯 Excellent win rate (>60%) - trading strategy is effective")
        elif win_rate > 45:
            recommendations.append("📊 Good win rate (45-60%) - strategy shows promise")
        elif win_rate > 30:
            recommendations.append("⚠️ Low win rate (30-45%) - consider strategy refinement")
        else:
            recommendations.append("❌ Very low win rate (<30%) - major strategy revision needed")

        # Trading frequency analysis
        if trade_frequency < 0.01:
            recommendations.append("📈 Very low trading frequency - model may be too conservative")
        elif trade_frequency > 0.5:
            recommendations.append("⚠️ High trading frequency - check for overtrading")
        else:
            recommendations.append("✅ Balanced trading frequency")

        # Portfolio performance
        if portfolio_return > 5:
            recommendations.append("📈 Strong portfolio returns - strategy is profitable")
        elif portfolio_return > 0:
            recommendations.append("� Positive returns - strategy is working")
        elif portfolio_return > -5:
            recommendations.append("⚠️ Small losses - minor adjustments needed")
        else:
            recommendations.append("❌ Significant losses - major strategy revision required")
