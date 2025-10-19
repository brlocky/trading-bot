"""
Training Log Visualizer for RL Models
Loads and visualizes stable-baselines3 training logs (CSV, Monitor, TensorBoard)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingLogVisualizer:
    """
    Visualizer for stable-baselines3 training logs
    Reads CSV, Monitor, and TensorBoard logs to create comprehensive training reports
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.progress_df = None
        self.monitor_df = None
        self.trading_df = None  # Add trading metrics DataFrame

    def load_logs(self):
        """Load all available log files"""
        print(f"📂 Loading logs from: {self.log_dir}")

        # Load progress.csv (main training metrics)
        progress_file = self.log_dir / "progress.csv"
        if progress_file.exists():
            self.progress_df = pd.read_csv(progress_file)
            print(f"✅ Loaded progress.csv: {len(self.progress_df)} entries")
        else:
            print("⚠️ progress.csv not found")

        # Load monitor.csv (episode rewards)
        monitor_file = self.log_dir / "monitor.csv"
        if monitor_file.exists():
            # Monitor CSV has special format - skip first line (metadata)
            self.monitor_df = pd.read_csv(monitor_file, skiprows=1)
            print(f"✅ Loaded monitor.csv: {len(self.monitor_df)} episodes")
        else:
            print("⚠️ monitor.csv not found")

        # Load trading_log.csv (custom trading metrics)
        trading_file = self.log_dir / "training_logs.csv"
        if trading_file.exists():
            self.trading_df = pd.read_csv(trading_file)
            print(f"✅ Loaded trading_logs.csv: {len(self.trading_df)} episodes")
        else:
            self.trading_df = None
            print("⚠️ trading_logs.csv not found")

        # Check for other log files
        csv_files = list(self.log_dir.glob("*.csv"))
        print(f"📋 Found {len(csv_files)} CSV files total: {[f.name for f in csv_files]}")

    def create_training_dashboard(self, save_plot: bool = True, show_plot: bool = True):
        """Create comprehensive training dashboard"""

        if self.progress_df is None and self.monitor_df is None:
            print("❌ No log data available. Run load_logs() first.")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # Plot 1: Training Progress (if progress.csv available)
        if self.progress_df is not None and not self.progress_df.empty:
            plt.subplot(3, 3, 1)
            has_data = False
            if 'rollout/ep_rew_mean' in self.progress_df.columns:
                plt.plot(self.progress_df['time/total_timesteps'],
                         self.progress_df['rollout/ep_rew_mean'], 'b-', linewidth=2, label='Mean Episode Reward')
                plt.fill_between(self.progress_df['time/total_timesteps'],
                                 self.progress_df['rollout/ep_rew_mean'] - self.progress_df.get('rollout/ep_rew_std', 0),
                                 self.progress_df['rollout/ep_rew_mean'] + self.progress_df.get('rollout/ep_rew_std', 0),
                                 alpha=0.2)
                has_data = True
            plt.title('Episode Reward Progress')
            plt.xlabel('Timesteps')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            if has_data:
                plt.legend()

            # Plot 2: Learning Progress
            plt.subplot(3, 3, 2)
            has_data = False
            if 'train/learning_rate' in self.progress_df.columns:
                plt.plot(self.progress_df['time/total_timesteps'],
                         self.progress_df['train/learning_rate'], 'g-', linewidth=2, label='Learning Rate')
                has_data = True
            plt.title('Learning Rate Schedule')
            plt.xlabel('Timesteps')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            if has_data:
                plt.legend()

            # Plot 3: Policy Loss
            plt.subplot(3, 3, 3)
            has_data = False
            if 'train/policy_gradient_loss' in self.progress_df.columns:
                plt.plot(self.progress_df['time/total_timesteps'],
                         self.progress_df['train/policy_gradient_loss'], 'r-', linewidth=2, label='Policy Loss')
                has_data = True
            plt.title('Policy Loss')
            plt.xlabel('Timesteps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            if has_data:
                plt.legend()

            # Plot 4: Value Loss
            plt.subplot(3, 3, 4)
            has_data = False
            if 'train/value_loss' in self.progress_df.columns:
                plt.plot(self.progress_df['time/total_timesteps'],
                         self.progress_df['train/value_loss'], 'orange', linewidth=2, label='Value Loss')
                has_data = True
            plt.title('Value Loss')
            plt.xlabel('Timesteps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            if has_data:
                plt.legend()

            # Plot 5: Explained Variance
            plt.subplot(3, 3, 5)
            has_data = False
            if 'train/explained_variance' in self.progress_df.columns:
                plt.plot(self.progress_df['time/total_timesteps'],
                         self.progress_df['train/explained_variance'], 'purple', linewidth=2, label='Explained Variance')
                has_data = True
            plt.title('Explained Variance')
            plt.xlabel('Timesteps')
            plt.ylabel('Explained Variance')
            plt.grid(True, alpha=0.3)
            if has_data:
                plt.legend()

            # Plot 6: Entropy
            plt.subplot(3, 3, 6)
            has_data = False
            if 'train/entropy_loss' in self.progress_df.columns:
                plt.plot(self.progress_df['time/total_timesteps'],
                         self.progress_df['train/entropy_loss'], 'brown', linewidth=2, label='Entropy')
                has_data = True
            plt.title('Entropy Loss')
            plt.xlabel('Timesteps')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)
            if has_data:
                plt.legend()

        # Plot 7: Episode Rewards from Monitor (if available)
        if self.monitor_df is not None and not self.monitor_df.empty:
            plt.subplot(3, 3, 7)
            plt.plot(range(len(self.monitor_df)), self.monitor_df['r'], 'teal', alpha=0.7, linewidth=1)
            # Add rolling average
            window = min(50, len(self.monitor_df) // 10)
            if window > 1:
                rolling_mean = self.monitor_df['r'].rolling(window=window).mean()
                plt.plot(range(len(self.monitor_df)), rolling_mean, 'red', linewidth=2, label=f'Rolling Mean ({window})')
                plt.legend()
            plt.title('Individual Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)

            # Plot 8: Episode Length
            plt.subplot(3, 3, 8)
            plt.plot(range(len(self.monitor_df)), self.monitor_df['l'], 'cyan', alpha=0.7, linewidth=1)
            plt.title('Episode Length')
            plt.xlabel('Episode')
            plt.ylabel('Length (steps)')
            plt.grid(True, alpha=0.3)

            # Plot 9: Episode Reward Distribution
            plt.subplot(3, 3, 9)
            plt.hist(self.monitor_df['r'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            plt.axvline(self.monitor_df['r'].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {self.monitor_df["r"].mean():.3f}')
            plt.title('Episode Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_file = self.log_dir / "training_dashboard.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 Training dashboard saved: {plot_file}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        if self.progress_df is not None and not self.progress_df.empty:
            print("MAIN TRAINING METRICS:")
            final_timesteps = self.progress_df['time/total_timesteps'].iloc[-1]
            print(f"   Total Timesteps: {final_timesteps:,}")

            if 'rollout/ep_rew_mean' in self.progress_df.columns:
                final_reward = self.progress_df['rollout/ep_rew_mean'].iloc[-1]
                best_reward = self.progress_df['rollout/ep_rew_mean'].max()
                print(f"   Final Mean Reward: {final_reward:.4f}")
                print(f"   Best Mean Reward: {best_reward:.4f}")

            if 'time/fps' in self.progress_df.columns:
                avg_fps = self.progress_df['time/fps'].mean()
                print(f"   Average FPS: {avg_fps:.1f}")

        if self.monitor_df is not None and not self.monitor_df.empty:
            try:
                print("\nEPISODE STATISTICS:")
                print(f"   Total Episodes: {len(self.monitor_df):,}")
                print(f"   Average Reward: {self.monitor_df['r'].mean():.4f}")
                print(f"   Reward Std: {self.monitor_df['r'].std():.4f}")
                print(f"   Best Episode: {self.monitor_df['r'].max():.4f}")
                print(f"   Worst Episode: {self.monitor_df['r'].min():.4f}")
                print(f"   Average Length: {self.monitor_df['l'].mean():.1f} steps")

                # Check if rewards are improving
                if len(self.monitor_df) > 20:
                    early_avg = self.monitor_df['r'].head(10).mean()
                    late_avg = self.monitor_df['r'].tail(10).mean()
                    improvement = late_avg - early_avg
                    print(f"   Improvement (last 10 vs first 10): {improvement:+.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating episode statistics: {e}")

        print("=" * 60)

    def print_available_metrics(self):
        """Show what metrics are available in the logs grouped by category"""
        print("AVAILABLE METRICS BY CATEGORY:")

        if self.progress_df is not None:
            print(f"\nProgress CSV ({len(self.progress_df)} entries):")

            # Group metrics by category
            time_metrics = [col for col in self.progress_df.columns if col.startswith('time/')]
            train_metrics = [col for col in self.progress_df.columns if col.startswith('train/')]
            rollout_metrics = [col for col in self.progress_df.columns if col.startswith('rollout/')]

            if time_metrics:
                print("  📊 TIME METRICS:")
                for col in time_metrics:
                    print(f"     - {col}")

            if train_metrics:
                print("  🎯 TRAINING METRICS:")
                for col in train_metrics:
                    print(f"     - {col}")

            if rollout_metrics:
                print("  🎮 ROLLOUT METRICS:")
                for col in rollout_metrics:
                    print(f"     - {col}")

        if self.trading_df is not None:
            print(f"\nTrading CSV ({len(self.trading_df)} episodes):")
            print("  💰 TRADING PERFORMANCE METRICS:")
            for col in self.trading_df.columns:
                print(f"     - {col}")

        if self.monitor_df is not None:
            print(f"\nMonitor CSV ({len(self.monitor_df)} episodes):")
            print("  📈 EPISODE METRICS:")
            for col in self.monitor_df.columns:
                print(f"     - {col}")

    def create_comprehensive_dashboard(self, save_plot: bool = True, show_plot: bool = True):
        """Create comprehensive dashboard with all available metrics grouped by category"""

        if self.progress_df is None or self.progress_df.empty:
            print("❌ No progress data available for comprehensive dashboard")
            return

        # Group metrics by category
        time_metrics = [col for col in self.progress_df.columns if col.startswith('time/')]
        train_metrics = [col for col in self.progress_df.columns if col.startswith('train/')]
        rollout_metrics = [col for col in self.progress_df.columns if col.startswith('rollout/')]

        # Calculate number of subplots needed
        num_plots = len(time_metrics) + len(train_metrics) + len(rollout_metrics)
        if self.monitor_df is not None:
            num_plots += 2  # Episode rewards and lengths
        if self.trading_df is not None:
            num_plots += 4  # Trading performance charts

        # Create figure with dynamic subplot arrangement
        cols = 3
        rows = (num_plots + cols - 1) // cols  # Ceiling division
        fig = plt.figure(figsize=(20, 5 * rows))

        plot_idx = 1

        # Plot TIME metrics
        print(f"📊 Plotting {len(time_metrics)} time metrics...")
        for metric in time_metrics:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(self.progress_df['time/total_timesteps'], self.progress_df[metric],
                     linewidth=2, label=metric.split('/')[-1])
            plt.title(f'Time: {metric.split("/")[-1].replace("_", " ").title()}')
            plt.xlabel('Timesteps')
            plt.ylabel(metric.split('/')[-1])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_idx += 1

        # Plot TRAINING metrics
        print(f"🎯 Plotting {len(train_metrics)} training metrics...")
        for metric in train_metrics:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(self.progress_df['time/total_timesteps'], self.progress_df[metric],
                     linewidth=2, label=metric.split('/')[-1])
            plt.title(f'Training: {metric.split("/")[-1].replace("_", " ").title()}')
            plt.xlabel('Timesteps')
            plt.ylabel(metric.split('/')[-1])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_idx += 1

        # Plot ROLLOUT metrics
        print(f"🎮 Plotting {len(rollout_metrics)} rollout metrics...")
        for metric in rollout_metrics:
            plt.subplot(rows, cols, plot_idx)
            if metric == 'rollout/ep_rew_mean' and 'rollout/ep_rew_std' in self.progress_df.columns:
                # Special handling for reward with std deviation
                plt.plot(self.progress_df['time/total_timesteps'], self.progress_df[metric],
                         'b-', linewidth=2, label='Mean Reward')
                plt.fill_between(self.progress_df['time/total_timesteps'],
                                 self.progress_df[metric] - self.progress_df.get('rollout/ep_rew_std', 0),
                                 self.progress_df[metric] + self.progress_df.get('rollout/ep_rew_std', 0),
                                 alpha=0.2, label='±1 Std')
            else:
                plt.plot(self.progress_df['time/total_timesteps'], self.progress_df[metric],
                         linewidth=2, label=metric.split('/')[-1])
            plt.title(f'Rollout: {metric.split("/")[-1].replace("_", " ").title()}')
            plt.xlabel('Timesteps')
            plt.ylabel(metric.split('/')[-1])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_idx += 1

        # Plot MONITOR metrics (if available)
        if self.monitor_df is not None and not self.monitor_df.empty:
            print("Plotting monitor metrics...")

            # Episode rewards over time
            plt.subplot(rows, cols, plot_idx)
            plt.plot(range(len(self.monitor_df)), self.monitor_df['r'], alpha=0.7, linewidth=1)
            if len(self.monitor_df) > 10:
                window = min(50, len(self.monitor_df) // 10)
                rolling_mean = self.monitor_df['r'].rolling(window=window).mean()
                plt.plot(range(len(self.monitor_df)), rolling_mean, 'red', linewidth=2,
                         label=f'Rolling Mean ({window})')
                plt.legend()
            plt.title('Monitor: Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plot_idx += 1

            # Episode lengths over time
            if plot_idx <= rows * cols:
                plt.subplot(rows, cols, plot_idx)
                plt.plot(range(len(self.monitor_df)), self.monitor_df['l'], 'green', alpha=0.7, linewidth=1)
                plt.title('Monitor: Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Length (steps)')
                plt.grid(True, alpha=0.3)
                plot_idx += 1

        # Plot TRADING metrics (if available)
        if self.trading_df is not None and not self.trading_df.empty:
            print("Plotting trading performance metrics...")

            # Portfolio value over episodes
            if plot_idx <= rows * cols:
                plt.subplot(rows, cols, plot_idx)
                plt.plot(self.trading_df['episode'], self.trading_df['final_balance'], 'b-', linewidth=2, marker='o')
                if 'total_return_pct' in self.trading_df.columns:
                    plt.axhline(y=1000000, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
                    plt.legend()
                plt.title('Trading: Portfolio Value')
                plt.xlabel('Episode')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True, alpha=0.3)
                plot_idx += 1

            # Win rate over episodes
            if plot_idx <= rows * cols and 'win_rate' in self.trading_df.columns:
                plt.subplot(rows, cols, plot_idx)
                plt.plot(self.trading_df['episode'], self.trading_df['win_rate'], 'purple', linewidth=2, marker='o')
                plt.title('Trading: Win Rate')
                plt.xlabel('Episode')
                plt.ylabel('Win Rate (%)')
                plt.grid(True, alpha=0.3)
                plot_idx += 1

            # Total trades per episode
            if plot_idx <= rows * cols and 'total_trades' in self.trading_df.columns:
                plt.subplot(rows, cols, plot_idx)
                plt.bar(self.trading_df['episode'], self.trading_df['total_trades'], alpha=0.7, color='orange')
                plt.title('Trading: Trades per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Number of Trades')
                plt.grid(True, alpha=0.3)
                plot_idx += 1

            # Return percentage
            if plot_idx <= rows * cols and 'total_return_pct' in self.trading_df.columns:
                plt.subplot(rows, cols, plot_idx)
                plt.plot(self.trading_df['episode'], self.trading_df['total_return_pct'], 'g-', linewidth=2, marker='o')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break Even')
                plt.title('Trading: Return Percentage')
                plt.xlabel('Episode')
                plt.ylabel('Return (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_idx += 1

        plt.tight_layout()

        if save_plot:
            plot_file = self.log_dir / "comprehensive_training_dashboard.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 Comprehensive dashboard saved: {plot_file}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig


def view_training_logs(log_dir: str):
    """Convenience function to quickly view training logs

    Args:
        log_dir: Directory containing the training logs
        comprehensive: If True, shows all metrics grouped by category
    """
    try:
        visualizer = TrainingLogVisualizer(log_dir)
        visualizer.load_logs()
        visualizer.print_available_metrics
        visualizer.print_training_summary()
        visualizer.create_comprehensive_dashboard()
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")


def view_trading_performance(log_dir: str):
    """Convenience function to view ONLY trading performance charts (separate from PPO training)

    Args:
        log_dir: Directory containing the training logs
    """
    visualizer = TrainingLogVisualizer(log_dir)
    visualizer.load_logs()

    if visualizer.trading_df is not None and not visualizer.trading_df.empty:
        print("📊 Creating Trading Performance Dashboard...")
        print("=" * 50)

        # Create dedicated trading charts
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        df = visualizer.trading_df

        # Create figure with trading-specific plots only
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Trading Performance Dashboard (Separate from PPO Training)', fontsize=16, fontweight='bold')

        # 1. Portfolio Value Over Episodes
        axes[0, 0].plot(df['episode'], df['final_balance'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].axhline(y=1000000, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
        axes[0, 0].set_title('Portfolio Value Over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # 2. Return Percentage Over Episodes
        if 'total_return_pct' in df.columns:
            axes[0, 1].plot(df['episode'], df['total_return_pct'], 'g-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break Even')
            axes[0, 1].set_title('Return Percentage Over Episodes')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Trades Per Episode
        if 'total_trades' in df.columns:
            axes[0, 2].bar(df['episode'], df['total_trades'], alpha=0.7, color='orange', width=0.6)
            axes[0, 2].set_title('Total Trades Per Episode')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Number of Trades')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Win Rate Over Episodes
        if 'win_rate' in df.columns:
            axes[1, 0].plot(df['episode'], df['win_rate'], 'purple', linewidth=2, marker='o', markersize=4)
            axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Win Rate')
            axes[1, 0].set_title('Win Rate Over Episodes')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Profitable vs Losing Trades (Stacked Bar)
        if 'profitable_trades' in df.columns:
            losing_trades = df['total_trades'] - df['profitable_trades'] if 'total_trades' in df.columns else 0
            axes[1, 1].bar(df['episode'], df['profitable_trades'],
                           alpha=0.8, color='green', label='Profitable', width=0.6)
            axes[1, 1].bar(df['episode'], losing_trades,
                           bottom=df['profitable_trades'], alpha=0.8, color='red', label='Losing', width=0.6)
            axes[1, 1].set_title('Profitable vs Losing Trades')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Number of Trades')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Average Trade PnL
        if 'avg_trade_pnl' in df.columns:
            axes[1, 2].plot(df['episode'], df['avg_trade_pnl'], 'brown', linewidth=2, marker='o', markersize=4)
            axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break Even')
            axes[1, 2].set_title('Average Trade PnL')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Avg PnL per Trade ($)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        # 7. Net PnL Over Episodes
        net_pnl = df['final_balance'] - 1000000  # Calculate net PnL
        axes[2, 0].plot(df['episode'], net_pnl, 'teal', linewidth=2, marker='o', markersize=4)
        axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break Even')
        axes[2, 0].set_title('Net PnL Over Episodes')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Net PnL ($)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # 8. Trading Activity
        if 'total_trades' in df.columns and 'episode_length' in df.columns:
            trades_per_step = df['total_trades'] / df['episode_length']
            axes[2, 1].plot(df['episode'], trades_per_step, 'cyan', linewidth=2, marker='o', markersize=4)
            axes[2, 1].set_title('Trading Frequency')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Trades per Step')
            axes[2, 1].grid(True, alpha=0.3)

        # 9. Episode Length vs Performance (if data available)
        if 'episode_length' in df.columns and 'total_return_pct' in df.columns and 'win_rate' in df.columns:
            scatter = axes[2, 2].scatter(df['episode_length'], df['total_return_pct'],
                                         c=df['win_rate'], cmap='RdYlGn', alpha=0.7, s=60)
            axes[2, 2].set_title('Episode Length vs Return % (colored by Win Rate)')
            axes[2, 2].set_xlabel('Episode Length (steps)')
            axes[2, 2].set_ylabel('Return (%)')
            axes[2, 2].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[2, 2], label='Win Rate (%)')
        else:
            axes[2, 2].text(0.5, 0.5, 'Episode Length\ndata not available',
                            ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Episode Length vs Performance')

        plt.tight_layout()

        # Save plot
        plot_file = visualizer.log_dir / "trading_performance_dashboard.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Trading performance dashboard saved: {plot_file}")

        plt.show()

        # Print trading summary
        print("\n📊 TRADING PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Episodes: {len(df)}")
        total_trades = 0
        if 'total_trades' in df.columns:
            total_trades = df['total_trades'].sum()
            print(f"Total Trades: {total_trades}")
        if 'profitable_trades' in df.columns and total_trades > 0:
            total_profitable = df['profitable_trades'].sum()
            win_rate = (total_profitable / total_trades * 100)
            print(f"Profitable Trades: {total_profitable}")
            print(f"Overall Win Rate: {win_rate:.1f}%")
        if 'total_return_pct' in df.columns:
            final_return = df['total_return_pct'].iloc[-1] if len(df) > 0 else 0
            best_return = df['total_return_pct'].max()
            print(f"Final Return: {final_return:.2f}%")
            print(f"Best Return: {best_return:.2f}%")

        return visualizer
    else:
        print("❌ No trading performance data found")
        print("Make sure trading_metrics.csv or training_log.csv exists in the log directory")
        return None
