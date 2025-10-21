import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsMonitor(BaseCallback):
    def __init__(self, log_path="training_logs.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_metrics = []
        self._current_episode_data = []
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._file = open(self.log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        # Enhanced headers for trading metrics
        self._writer.writerow([
            "episode", "total_reward", "episode_length", "final_balance",
            "total_return_pct", "total_trades", "profitable_trades",
            "win_rate", "max_position", "avg_trade_pnl", "final_position"
        ])

    def _on_step(self) -> bool:
        if not self._writer:
            raise RuntimeError("TradingMetricsCallback not properly initialized.")

        # Collect step data for analysis
        infos = self.locals.get("infos", [])

        for idx, done in enumerate(self.locals["dones"]):
            # Initialize episode data if needed
            if len(self._current_episode_data) <= idx:
                self._current_episode_data.append({
                    'rewards': [],
                    'trades': [],
                    'positions': [],
                    'balances': [],
                    'step_count': 0
                })

            # Collect step metrics
            episode_data = self._current_episode_data[idx]
            episode_data['rewards'].append(self.locals["rewards"][idx])
            episode_data['step_count'] += 1

            # Extract trading info if available
            if idx < len(infos) and infos[idx]:
                info = infos[idx]
                episode_data['balances'].append(info.get('balance', 0))
                episode_data['positions'].append(info.get('position', 0))
                if info.get('traded', False):
                    episode_data['trades'].append(info.get('step_pnl', 0))

            if done:
                # Calculate episode metrics
                rewards = episode_data['rewards']
                trades = episode_data['trades']
                balances = episode_data['balances']
                positions = episode_data['positions']

                total_reward = sum(rewards)
                episode_length = episode_data['step_count']
                final_balance = balances[-1] if balances else 0
                total_return_pct = ((final_balance / 1000000) - 1) * 100 if balances else 0
                total_trades = len(trades)
                profitable_trades = sum(1 for t in trades if t > 0)
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                max_position = max(abs(p) for p in positions) if positions else 0
                avg_trade_pnl = np.mean(trades) if trades else 0
                final_position = positions[-1] if positions else 0

                # Write episode metrics
                self._writer.writerow([
                    len(self.episode_metrics) + 1, total_reward, episode_length,
                    final_balance, total_return_pct, total_trades, profitable_trades,
                    win_rate, max_position, avg_trade_pnl, final_position
                ])

                # IMPORTANT: Flush data immediately to ensure it's written
                if self._file:
                    self._file.flush()

                if self.verbose > 0:
                    print(f"📊 Episode {len(self.episode_metrics) + 1} completed: "
                          f"Return={total_return_pct:.2f}%, Trades={total_trades}, "
                          f"Win Rate={win_rate:.1f}%, Balance=${final_balance:,.0f}")

                # Store for later analysis - KEEP ALL TRADING METRICS
                self.episode_metrics.append({
                    'episode_number': len(self.episode_metrics) + 1,
                    'total_reward': total_reward,
                    'episode_length': episode_length,
                    'final_balance': final_balance,
                    'total_return_pct': total_return_pct,
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'win_rate': win_rate,
                    'max_position': max_position,
                    'avg_trade_pnl': avg_trade_pnl,
                    'final_position': final_position,
                    # Additional derived metrics for charting
                    'losing_trades': total_trades - profitable_trades,
                    'initial_balance': 1000000,  # Store initial balance for calculations
                    'net_pnl': final_balance - 1000000,
                    'trades_per_step': total_trades / episode_length if episode_length > 0 else 0
                })

                # Reset for next episode
                self._current_episode_data[idx] = {
                    'rewards': [], 'trades': [], 'positions': [],
                    'balances': [], 'step_count': 0
                }

        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()
            print(f"✅ Training metrics saved to {self.log_path}")

        # Print summary
        if self.episode_metrics:
            avg_return = np.mean([ep['total_return_pct'] for ep in self.episode_metrics])
            avg_trades = np.mean([ep['total_trades'] for ep in self.episode_metrics])
            avg_win_rate = np.mean([ep['win_rate'] for ep in self.episode_metrics])
            total_episodes = len(self.episode_metrics)
            total_trades_all = sum([ep['total_trades'] for ep in self.episode_metrics])
            total_profitable = sum([ep['profitable_trades'] for ep in self.episode_metrics])

            print("Trading Summary:")
            print(f"   Episodes: {total_episodes}")
            print(f"   Total Trades: {total_trades_all}")
            print(f"   Profitable Trades: {total_profitable}")
            print(f"   Overall Win Rate: {(total_profitable/total_trades_all*100) if total_trades_all > 0 else 0:.1f}%")
            print(f"   Avg Return per Episode: {avg_return:.2f}%")
            print(f"   Avg Trades per Episode: {avg_trades:.1f}")
            print(f"   Avg Win Rate per Episode: {avg_win_rate:.1f}%")
