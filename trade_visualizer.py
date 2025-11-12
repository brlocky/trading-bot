"""
Trade Visualizer - Analyze and label trades from training logs
"""
from features.visible_range_vp import VisibleRangeVP
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.append('src')


class TradeVisualizer:
    """
    Interactive visualizer for labeling trades based on VP context.
    Shows chart with VP levels, entry/exit points, and allows labeling.
    """

    def __init__(self, data_path: str, trades_path: str, lookback: int = 288):
        """
        Args:
            data_path: Path to PKL file with OHLCV data
            trades_path: Path to PKL file with trade logs
            lookback: Lookback window for VP calculation
        """
        # Load market data
        self.df = pd.read_pickle(data_path)
        print(f"✓ Loaded {len(self.df):,} candles")

        # Load trades
        with open(trades_path, 'rb') as f:
            self.trades = pickle.load(f)
        print(f"✓ Loaded {len(self.trades)} trades")

        self.lookback = lookback
        self.vp = VisibleRangeVP(n_bins=50)

        # Current trade index
        self.current_idx = 0

        # Label mapping
        self.labels = {
            'good_entry': 0,
            'bad_entry': 0,
            'neutral': 0,
            'skip': 0
        }

    def _calculate_vp_for_trade(self, timestamp: int) -> Dict:
        """Calculate VP levels for trade context."""
        start_idx = max(0, timestamp - self.lookback)
        window_data = self.df.iloc[start_idx:timestamp]

        if len(window_data) == 0:
            return None

        _, levels = self.vp.calculate_vp(window_data)
        return levels

    def _plot_trade(self, trade: Dict, ax):
        """Plot trade with VP context."""
        timestamp = trade['timestamp']

        # Get window around trade
        start = max(0, timestamp - 100)
        end = min(len(self.df), timestamp + 100)
        window = self.df.iloc[start:end]

        # Calculate VP at entry
        vp_levels = self._calculate_vp_for_trade(timestamp)

        # Plot candlesticks
        for i, (idx, row) in enumerate(window.iterrows()):
            color = 'green' if row['close'] > row['open'] else 'red'
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.5)
            ax.plot([i, i], [row['open'], row['close']], color=color, linewidth=2)

        # Plot VP levels
        if vp_levels:
            entry_idx = timestamp - start
            ax.axhline(vp_levels['vah'], color='blue', linestyle='--', alpha=0.7, label='VAH')
            ax.axhline(vp_levels['poc'], color='yellow', linestyle='-', alpha=0.9, linewidth=2, label='POC')
            ax.axhline(vp_levels['val'], color='blue', linestyle='--', alpha=0.7, label='VAL')
            ax.axhline(vp_levels['high'], color='gray', linestyle=':', alpha=0.5, label='Range High')
            ax.axhline(vp_levels['low'], color='gray', linestyle=':', alpha=0.5, label='Range Low')

            # Fill Value Area
            ax.fill_between([0, len(window)], vp_levels['val'], vp_levels['vah'],
                            alpha=0.1, color='blue', label='Value Area')

        # Mark entry
        entry_idx = timestamp - start
        entry_price = trade['entry_price']
        ax.scatter(entry_idx, entry_price, color='lime' if trade['action'] == 'LONG' else 'red',
                   marker='^' if trade['action'] == 'LONG' else 'v', s=200, zorder=5,
                   edgecolors='black', linewidths=2, label=f"{trade['action']} Entry")

        # Mark exit if closed
        if trade.get('exit_price'):
            exit_idx = trade.get('exit_timestamp', entry_idx + trade.get('duration', 10))
            exit_idx = exit_idx - start
            exit_price = trade['exit_price']

            # Draw P&L line
            pnl_color = 'green' if trade['pnl'] > 0 else 'red'
            ax.plot([entry_idx, exit_idx], [entry_price, exit_price],
                    color=pnl_color, linewidth=2, linestyle='--', alpha=0.7)
            ax.scatter(exit_idx, exit_price, color=pnl_color, marker='X', s=200,
                       zorder=5, edgecolors='black', linewidths=2, label='Exit')

        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Trade #{self.current_idx + 1}/{len(self.trades)}")

        # Add trade info text
        info_text = f"{trade['action']} @ ${entry_price:.2f}\n"
        if trade.get('exit_price'):
            info_text += f"Exit @ ${trade['exit_price']:.2f}\n"
            info_text += f"P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)\n"
            info_text += f"Duration: {trade['duration']} candles"

        # Add VP context
        if vp_levels:
            info_text += f"\n\nVP Context at Entry:"
            info_text += f"\nDist to VAH: {trade.get('dist_to_vah', 0):.3f}"
            info_text += f"\nDist to POC: {trade.get('dist_to_poc', 0):.3f}"
            info_text += f"\nDist to VAL: {trade.get('dist_to_val', 0):.3f}"
            info_text += f"\n\nIn VA: {trade.get('close_in_va', False)}"
            info_text += f"\nAbove POC: {trade.get('close_above_poc', False)}"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _label_trade(self, label: str):
        """Label current trade and move to next."""
        if self.current_idx >= len(self.trades):
            print("No more trades to label!")
            return

        # Save label
        self.trades[self.current_idx]['label'] = label
        self.labels[label] += 1

        print(f"✓ Labeled trade #{self.current_idx + 1} as: {label}")

        # Move to next trade
        self.current_idx += 1

        if self.current_idx < len(self.trades):
            self._show_next_trade()
        else:
            print("\n🎉 Labeling complete!")
            self._show_summary()
            plt.close()

    def _show_next_trade(self):
        """Display next trade for labeling."""
        plt.clf()

        fig = plt.gcf()

        # Main plot
        ax = plt.subplot(111)
        self._plot_trade(self.trades[self.current_idx], ax)

        # Add buttons
        button_height = 0.05
        button_width = 0.15
        button_y = 0.02

        ax_good = plt.axes([0.1, button_y, button_width, button_height])
        ax_bad = plt.axes([0.3, button_y, button_width, button_height])
        ax_neutral = plt.axes([0.5, button_y, button_width, button_height])
        ax_skip = plt.axes([0.7, button_y, button_width, button_height])

        btn_good = Button(ax_good, 'Good Entry', color='lightgreen')
        btn_bad = Button(ax_bad, 'Bad Entry', color='lightcoral')
        btn_neutral = Button(ax_neutral, 'Neutral', color='lightgray')
        btn_skip = Button(ax_skip, 'Skip', color='lightyellow')

        btn_good.on_clicked(lambda event: self._label_trade('good_entry'))
        btn_bad.on_clicked(lambda event: self._label_trade('bad_entry'))
        btn_neutral.on_clicked(lambda event: self._label_trade('neutral'))
        btn_skip.on_clicked(lambda event: self._label_trade('skip'))

        plt.draw()

    def _show_summary(self):
        """Show labeling summary."""
        print("\n" + "="*60)
        print("LABELING SUMMARY")
        print("="*60)
        print(f"Good Entries:   {self.labels['good_entry']}")
        print(f"Bad Entries:    {self.labels['bad_entry']}")
        print(f"Neutral:        {self.labels['neutral']}")
        print(f"Skipped:        {self.labels['skip']}")
        print(f"Total:          {sum(self.labels.values())}")
        print("="*60)

    def start_labeling(self):
        """Start interactive labeling session."""
        if len(self.trades) == 0:
            print("No trades to label!")
            return

        print("\n" + "="*60)
        print("TRADE LABELING SESSION")
        print("="*60)
        print("Instructions:")
        print("  - Click 'Good Entry' if entry was at a good VP level")
        print("    (e.g., bought near VAL/POC support, sold near VAH/POC resistance)")
        print("  - Click 'Bad Entry' if entry was at a bad level")
        print("    (e.g., bought at VAH resistance, sold at VAL support)")
        print("  - Click 'Neutral' if trade doesn't show clear VP context")
        print("  - Click 'Skip' to skip this trade")
        print("="*60 + "\n")

        plt.figure(figsize=(14, 8))
        self._show_next_trade()
        plt.show()

    def save_labeled_trades(self, output_path: str):
        """Save labeled trades to new PKL file."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.trades, f)
        print(f"💾 Saved labeled trades to {output_path}")

    def analyze_labels(self) -> Dict:
        """Analyze labeled trades for patterns."""
        labeled = [t for t in self.trades if t.get('label') and t['label'] != 'skip']

        if len(labeled) == 0:
            print("No labeled trades to analyze!")
            return {}

        # Good entries stats
        good_entries = [t for t in labeled if t['label'] == 'good_entry']
        bad_entries = [t for t in labeled if t['label'] == 'bad_entry']

        analysis = {
            'total_labeled': len(labeled),
            'good_entries': len(good_entries),
            'bad_entries': len(bad_entries),
        }

        # Analyze good entries
        if good_entries:
            good_long = [t for t in good_entries if t['action'] == 'LONG']
            good_short = [t for t in good_entries if t['action'] == 'SHORT']

            analysis['good_long_avg_dist_to_val'] = np.mean([t.get('dist_to_val', 0) for t in good_long]) if good_long else 0
            analysis['good_long_avg_dist_to_poc'] = np.mean([t.get('dist_to_poc', 0) for t in good_long]) if good_long else 0
            analysis['good_short_avg_dist_to_vah'] = np.mean([t.get('dist_to_vah', 0) for t in good_short]) if good_short else 0
            analysis['good_short_avg_dist_to_poc'] = np.mean([t.get('dist_to_poc', 0) for t in good_short]) if good_short else 0

        return analysis


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize and label trades')
    parser.add_argument('--data', type=str, default='data/binance-BTCUSDT-5m.pkl',
                        help='Path to market data PKL')
    parser.add_argument('--trades', type=str, required=True,
                        help='Path to trades PKL file')
    parser.add_argument('--output', type=str, default='logs/trades/trades_labeled.pkl',
                        help='Output path for labeled trades')

    args = parser.parse_args()

    # Create visualizer
    viz = TradeVisualizer(args.data, args.trades)

    # Start labeling
    viz.start_labeling()

    # Save results
    viz.save_labeled_trades(args.output)

    # Analyze
    analysis = viz.analyze_labels()
    print("\nAnalysis:", analysis)
