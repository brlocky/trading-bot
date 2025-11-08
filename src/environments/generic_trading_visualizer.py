import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import imageio.v3 as iio
from typing import List, Dict, Tuple, Any


class GenericTradingVisualizer:
    """
    A generic, reusable class for creating trading visualization dashboards.
    """

    def __init__(self,
                 subplot_config: List[Dict] = None,
                 figsize: Tuple[int, int] = (12, 10),
                 style: str = 'default'):
        self.figsize = figsize
        self.style = style

        # Default subplot configuration
        self.subplot_config = subplot_config or self._get_default_config()

        # Color schemes
        self.colors = {
            'price': 'tab:blue',
            'volume': 'tab:orange',
            'long': 'green',
            'short': 'red',
            'exit': 'black',
            'tp': 'green',
            'sl': 'red',
            'equity': 'tab:orange',
            'position': 'tab:purple',
            'reward': 'tab:cyan',
            'pnl': 'tab:pink'
        }

    def _get_default_config(self) -> List[Dict]:
        """Get default subplot configuration."""
        return [
            {
                'name': 'price',
                'title': 'Price & Trades',
                'height_ratio': 3,
                'plots': ['price', 'trades', 'volume_profile', 'pivots'],
                'y_label': 'Price'
            },
            {
                'name': 'position',
                'title': 'Position Size',
                'height_ratio': 1,
                'plots': ['position_size'],
                'y_label': 'Position/Balance'
            },
            {
                'name': 'equity',
                'title': 'Account Equity',
                'height_ratio': 1,
                'plots': ['equity'],
                'y_label': 'Equity'
            },
            {
                'name': 'reward',
                'title': 'Reward',
                'height_ratio': 1,
                'plots': ['reward'],
                'y_label': 'Reward'
            },
            {
                'name': 'pnl',
                'title': 'Realized PnL',
                'height_ratio': 1,
                'plots': ['pnl'],
                'y_label': 'PnL'
            }
        ]

    def create_figure(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create figure with configured subplots."""
        plt.style.use(self.style)

        height_ratios = [config['height_ratio'] for config in self.subplot_config]

        fig, axes = plt.subplots(
            len(self.subplot_config), 1,
            figsize=self.figsize,
            gridspec_kw={'height_ratios': height_ratios},
            sharex=True
        )

        # Ensure axes is always a list
        if len(self.subplot_config) == 1:
            axes = [axes]

        return fig, axes

    def plot_data(self,
                  data: Dict[str, Any],
                  trade_history: List[Dict],
                  indicators: Dict[str, Any] = None,
                  current_step: int = 0,
                  lookback_window: int = 100,
                  title: str = "Trading Dashboard") -> np.ndarray:
        """
        Main method to create the complete visualization.
        """
        # Calculate data range
        start = max(0, current_step - lookback_window)
        end = current_step

        # Create figure
        fig, axes = self.create_figure()

        # Plot each subplot according to configuration
        for i, (config, ax) in enumerate(zip(self.subplot_config, axes)):
            self._plot_subplot(ax, config, data, trade_history, indicators, start, end)

        # Set overall title and layout
        performance_stats = self._calculate_performance(trade_history)
        fig.suptitle(f"{title} | Step {current_step} | {performance_stats}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Convert to image
        return self._figure_to_array(fig)

    def _plot_subplot(self,
                      ax: plt.Axes,
                      config: Dict,
                      data: Dict[str, Any],
                      trade_history: List[Dict],
                      indicators: Dict[str, Any],
                      start: int,
                      end: int):
        """Plot a single subplot based on configuration."""
        ax.set_title(config.get('title', ''))
        ax.set_ylabel(config.get('y_label', ''))

        # Plot each specified plot type
        for plot_type in config.get('plots', []):
            plot_method = getattr(self, f'_plot_{plot_type}', None)
            if plot_method:
                # Pass all required parameters explicitly
                if plot_type == 'price':
                    plot_method(ax, data, start, end)
                elif plot_type == 'trades':
                    plot_method(ax, data, trade_history, start, end)
                elif plot_type == 'volume_profile':
                    plot_method(ax, data, indicators, start, end)
                elif plot_type == 'position_size':
                    plot_method(ax, trade_history, start, end)
                elif plot_type == 'equity':
                    plot_method(ax, trade_history, start, end)
                elif plot_type == 'reward':
                    plot_method(ax, indicators, start, end)
                elif plot_type == 'pnl':
                    plot_method(ax, trade_history, start, end)
                elif plot_type == 'pivots':
                    plot_method(ax, data, start, end)
        # Add grid
        ax.grid(True, alpha=0.3)

        # Add legend for the main price plot
        if config['name'] == 'price':
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                      loc='upper left', bbox_to_anchor=(0.01, 0.99),
                      fontsize=8)

    def _plot_price(self, ax: plt.Axes, data: Dict, start: int, end: int):
        """Plot price data."""
        if 'close' in data:
            close = data['close'].iloc[start:end].values
            ax.plot(close, label='Close Price', color=self.colors['price'], linewidth=1.5)

    def _plot_trades(self, ax: plt.Axes, data: Dict, trade_history: List[Dict], start: int, end: int):
        """Plot trade entries, exits, TP, and SL."""

        # Track which labels we've already added to legend
        labels_added = set()
        trades = trade_history[-1]['trades']

        for trade in trades:
            # Plot entry
            step_open = trade.get('step_open')
            if step_open is not None:
                self._plot_single_trade_entry(ax, trade, step_open - start, labels_added, start, end)

            # Plot exit if trade is closed
            if trade.get('status') == 'CLOSED':
                step_close = trade.get('step_close')
                if step_close is not None:
                    self._plot_single_trade_exit(ax, trade, step_close - start, labels_added, start, end)

    def _plot_single_trade_entry(self, ax: plt.Axes, trade: Dict, rel_step: int, labels_added: set, start: int, end: int):
        """Plot a single trade entry with markers."""

        # Check if this step is within visible range
        absolute_step = rel_step + start
        if not (start <= absolute_step < end):
            return

        position_size = trade.get('position_size', 0)
        entry_price = trade.get('entry_price', 0)

        if position_size > 0:  # Long entry
            label = 'Long Entry' if 'Long Entry' not in labels_added else ""
            ax.scatter(rel_step, entry_price, color=self.colors['long'], marker='^', s=100, zorder=5, label=label)
            labels_added.add('Long Entry')

            # Draw entry line for better visibility
            ax.axvline(x=rel_step, color=self.colors['long'], alpha=0.3, linestyle='--')

        elif position_size < 0:  # Short entry
            label = 'Short Entry' if 'Short Entry' not in labels_added else ""
            ax.scatter(rel_step, entry_price, color=self.colors['short'], marker='v', s=100, zorder=5, label=label)
            labels_added.add('Short Entry')

            # Draw entry line for better visibility
            ax.axvline(x=rel_step, color=self.colors['short'], alpha=0.3, linestyle='--')

    def _plot_single_trade_exit(self, ax: plt.Axes, trade: Dict, rel_step: int, labels_added: set, start: int, end: int):
        """Plot a single trade exit with markers and annotations."""

        # Check if this step is within visible range
        absolute_step = rel_step + start
        if not (start <= absolute_step < end):
            return

        exit_price = trade.get('exit_price', 0)

        # Plot exit marker
        label = 'Exit' if 'Exit' not in labels_added else ""
        ax.scatter(rel_step, exit_price, color=self.colors['exit'], marker='x', s=100, zorder=5, label=label)
        labels_added.add('Exit')

        # Draw exit line for better visibility
        ax.axvline(x=rel_step, color=self.colors['exit'], alpha=0.3, linestyle=':')

        # Annotate PnL for exits
        pnl = trade.get('pnl', 0)
        if pnl != 0:
            color = 'green' if pnl > 0 else 'red'
            # Position the text above or below the point
            va = 'bottom' if pnl > 0 else 'top'
            offset = 0.002 * exit_price if pnl > 0 else -0.002 * exit_price
            ax.text(rel_step, exit_price + offset, f"{pnl:.2f}", color=color,
                    fontsize=8, ha='center', va=va, weight='bold')

    def _plot_pivots(self, ax: plt.Axes, data: Dict, start: int, end: int):
        """Plot HH, LH, LL, HL pivots on the chart."""

        # Track which labels we've already added to legend
        labels_added = set()

        # Plot each pivot type
        pivot_config = {
            'HH': {'marker': '^', 'color': 'red', 'size': 80},
            'LH': {'marker': 'v', 'color': 'blue', 'size': 60},
            'LL': {'marker': 'v', 'color': 'green', 'size': 80},
            'HL': {'marker': '^', 'color': 'orange', 'size': 60}
        }

        for pivot_type, config in pivot_config.items():
            if pivot_type in data and not data[pivot_type].empty:
                self._plot_single_pivot_type(ax, data[pivot_type], pivot_type,
                                             config, labels_added, start, end)

    def _plot_single_pivot_type(self, ax: plt.Axes, pivot_values: List, pivot_type: str,
                                config: Dict, labels_added: set, start: int, end: int):
        """Plot a single type of pivot points."""

        marker = config['marker']
        color = config['color']
        size = config['size']

        for pivot in pivot_values:
            # pivot should be a tuple/dict with (step, price) or similar structure
            if isinstance(pivot, (tuple, list)) and len(pivot) >= 2:
                step, price = pivot[0], pivot[1]
            elif isinstance(pivot, dict):
                step = pivot.get('step', pivot.get('index', 0))
                price = pivot.get('price', pivot.get('value', 0))
            else:
                continue

            # Check if pivot is within visible range
            if start <= step < end:
                rel_step = step - start

                # Create label only if not already added to legend
                label = pivot_type if pivot_type not in labels_added else ""
                ax.scatter(rel_step, price, marker=marker, color=color, s=size,
                           zorder=6, label=label, edgecolors='black', linewidth=1)

                # Add pivot type text annotation
                ax.text(rel_step, price, pivot_type, fontsize=8, ha='center',
                        va='bottom' if marker == 'v' else 'top', weight='bold',
                        color=color, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

                labels_added.add(pivot_type)

    def _plot_volume_profile(self, ax: plt.Axes, data: Dict, indicators: Dict, start: int, end: int):
        """Plot volume profile with VAH, VAL, POC, and naked POC levels."""
        # Check if volume profile object is available
        vp = indicators.get('volume_profile') if indicators else None

        if vp is None:
            return

        # Extract cumulative weights and bins from the volume profile object (all-time volume)
        vp_weights = getattr(vp, 'cumulative_weights', None)
        vp_bins = getattr(vp, 'cumulative_bins', None)

        if vp_weights is not None and vp_bins is not None:
            if hasattr(vp_weights, 'cpu'):  # If it's a tensor
                vp_weights = vp_weights.cpu().numpy()
            if hasattr(vp_bins, 'cpu'):  # If it's a tensor
                vp_bins = vp_bins.cpu().numpy()

            bin_edges = vp_bins
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_height = bin_edges[1] - bin_edges[0]

            close_length = len(data['close'].iloc[start:end]) if 'close' in data else 100
            if vp_weights.max() > 0:
                vp_width = vp_weights / vp_weights.max() * close_length * 0.3
            else:
                vp_width = vp_weights

            ax.barh(bin_centers, vp_width, height=bin_height*0.9, alpha=0.3,
                    color='tab:orange', label="Volume Profile")

            # Calculate current session VAH, VAL, POC
            current_vah, current_val, current_poc = None, None, None
            if hasattr(vp, 'current_session_prices') and vp.current_session_idx > 0:
                prices = vp.current_session_prices[:vp.current_session_idx]
                volumes = vp.current_session_volumes[:vp.current_session_idx]
                if len(prices) > 0:
                    current_vah, current_val, current_poc = vp._calculate_value_area_fast(prices, volumes)

            # Plot current session VAH, VAL, POC
            if current_vah is not None:
                ax.axhline(current_vah, color='green', linestyle='-', linewidth=2, alpha=0.8, label='VAH')
                ax.text(close_length * 0.02, current_vah, 'VAH', color='green', fontsize=9,
                        weight='bold', va='bottom', bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='white', alpha=0.8))

            if current_val is not None:
                ax.axhline(current_val, color='red', linestyle='-', linewidth=2, alpha=0.8, label='VAL')
                ax.text(close_length * 0.02, current_val, 'VAL', color='red', fontsize=9,
                        weight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3',
                                                           facecolor='white', alpha=0.8))

            if current_poc is not None:
                ax.axhline(current_poc, color='blue', linestyle='-', linewidth=2.5, alpha=0.8, label='POC')
                ax.text(close_length * 0.02, current_poc, 'POC', color='blue', fontsize=10,
                        weight='bold', va='center', bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='yellow', alpha=0.8))

            # Plot previous day levels (dotted lines, lighter colors)
            if hasattr(vp, 'prev_day_vah') and vp.prev_day_vah is not None:
                ax.axhline(vp.prev_day_vah, color='lightgreen', linestyle=':', linewidth=1.5, alpha=0.6, label='Prev Day VAH')
                ax.text(close_length * 0.15, vp.prev_day_vah, 'PD-VAH', color='green', fontsize=7, ha='left', va='bottom')

            if hasattr(vp, 'prev_day_val') and vp.prev_day_val is not None:
                ax.axhline(vp.prev_day_val, color='lightcoral', linestyle=':', linewidth=1.5, alpha=0.6, label='Prev Day VAL')
                ax.text(close_length * 0.15, vp.prev_day_val, 'PD-VAL', color='red', fontsize=7, ha='left', va='top')

            if hasattr(vp, 'prev_day_poc') and vp.prev_day_poc is not None:
                ax.axhline(vp.prev_day_poc, color='lightblue', linestyle=':', linewidth=2, alpha=0.6, label='Prev Day POC')
                ax.text(close_length * 0.15, vp.prev_day_poc, 'PD-POC', color='blue', fontsize=8, ha='left', va='center')

            if hasattr(vp, 'prev_day_high') and vp.prev_day_high is not None:
                ax.axhline(vp.prev_day_high, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(close_length * 0.25, vp.prev_day_high, 'PD-High', color='gray', fontsize=6, ha='left', va='bottom')

            if hasattr(vp, 'prev_day_low') and vp.prev_day_low is not None:
                ax.axhline(vp.prev_day_low, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(close_length * 0.25, vp.prev_day_low, 'PD-Low', color='gray', fontsize=6, ha='left', va='top')

            if hasattr(vp, 'prev_day_close') and vp.prev_day_close is not None:
                ax.axhline(vp.prev_day_close, color='darkgray', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(close_length * 0.25, vp.prev_day_close, 'PD-Close', color='darkgray', fontsize=6, ha='left', va='center')

            # Plot previous week levels (dashed lines, even lighter)
            if hasattr(vp, 'prev_week_vah') and vp.prev_week_vah is not None:
                ax.axhline(vp.prev_week_vah, color='palegreen', linestyle='--', linewidth=1.5, alpha=0.5, label='Prev Week VAH')
                ax.text(close_length * 0.35, vp.prev_week_vah, 'PW-VAH', color='green', fontsize=6, ha='left', va='bottom')

            if hasattr(vp, 'prev_week_val') and vp.prev_week_val is not None:
                ax.axhline(vp.prev_week_val, color='mistyrose', linestyle='--', linewidth=1.5, alpha=0.5, label='Prev Week VAL')
                ax.text(close_length * 0.35, vp.prev_week_val, 'PW-VAL', color='red', fontsize=6, ha='left', va='top')

            if hasattr(vp, 'prev_week_poc') and vp.prev_week_poc is not None:
                ax.axhline(vp.prev_week_poc, color='lightsteelblue', linestyle='--', linewidth=2, alpha=0.5, label='Prev Week POC')
                ax.text(close_length * 0.35, vp.prev_week_poc, 'PW-POC', color='blue', fontsize=7, ha='left', va='center')

            # Plot naked POCs
            if hasattr(vp, 'naked_daily_pocs') and vp.naked_daily_pocs:
                for i, naked_poc in enumerate(vp.naked_daily_pocs):
                    if isinstance(naked_poc, dict):
                        price = naked_poc.get('poc')
                    elif isinstance(naked_poc, (tuple, list)) and len(naked_poc) > 1:
                        price = naked_poc[1]
                    else:
                        continue

                    if price is not None:
                        label = 'Naked Daily POC' if i == 0 else ""
                        ax.axhline(price, color='purple', linestyle=':', linewidth=1.5,
                                   alpha=0.6, label=label)
                        ax.text(close_length * 0.98, price, f'D-POC', color='purple',
                                fontsize=8, weight='bold', ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='lavender', alpha=0.7))

            if hasattr(vp, 'naked_weekly_pocs') and vp.naked_weekly_pocs:
                for i, naked_poc in enumerate(vp.naked_weekly_pocs):
                    if isinstance(naked_poc, dict):
                        price = naked_poc.get('poc')
                    elif isinstance(naked_poc, (tuple, list)) and len(naked_poc) > 1:
                        price = naked_poc[1]
                    else:
                        continue

                    if price is not None:
                        label = 'Naked Weekly POC' if i == 0 else ""
                        ax.axhline(price, color='darkmagenta', linestyle='-.', linewidth=2,
                                   alpha=0.7, label=label)
                        ax.text(close_length * 0.98, price, f'W-POC', color='darkmagenta',
                                fontsize=9, weight='bold', ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='plum', alpha=0.7))

    def _plot_position_size(self, ax: plt.Axes, trade_history: List[Dict], start: int, end: int):
        """Plot position size over time."""
        relevant_trades = [t for t in trade_history if start <= t.get('step', 0) < end]
        if not relevant_trades:
            ax.set_visible(False)
            return

        steps = [t['step'] - start for t in relevant_trades]
        positions = [t.get('position_size', 0) for t in relevant_trades]

        ax.plot(steps, positions, color=self.colors['position'], linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('Position Size')

    def _plot_equity(self, ax: plt.Axes, trade_history: List[Dict], start: int, end: int):
        """Plot equity curve."""
        relevant_trades = [t for t in trade_history if start <= t.get('step', 0) < end]
        if not relevant_trades:
            ax.set_visible(False)
            return

        steps = [t['step'] - start for t in relevant_trades]
        equity = [t.get('equity', 0) for t in relevant_trades]

        ax.plot(steps, equity, color=self.colors['equity'], linewidth=1.5)
        ax.set_ylabel('Equity')

    def _plot_reward(self, ax: plt.Axes, indicators: Dict, start: int, end: int):
        """Plot reward over time."""
        # Check if history data is available in indicators
        if indicators and 'history_data' in indicators:
            history_data = indicators['history_data']
            if history_data:
                # Filter history data for the current window
                hist_slice = [h for h in history_data if start <= h.get('step', 0) < end]

                if hist_slice:
                    steps = [h['step'] - start for h in hist_slice]
                    rewards = [h.get('reward', 0) for h in hist_slice]

                    ax.plot(steps, rewards, color=self.colors['reward'], linewidth=1.5)
                    ax.set_ylabel('Reward')
                    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
                    return

        # If no reward data, hide the subplot
        ax.set_visible(False)

    def _plot_pnl(self, ax: plt.Axes, trade_history: List[Dict], start: int, end: int):
        """Plot realized PnL over time."""
        relevant_trades = [t for t in trade_history if start <= t.get('step', 0) < end]
        if not relevant_trades:
            ax.set_visible(False)
            return

        steps = [t['step'] - start for t in relevant_trades]
        pnls = [t.get('realized_pnl', 0) for t in relevant_trades]

        ax.plot(steps, pnls, color=self.colors['pnl'], linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('Realized PnL')

    def _calculate_performance(self, trade_history: List[Dict]) -> str:
        """Calculate and format performance statistics."""
        if not trade_history:
            return "No trades"

        total_pnl = sum(t.get('realized_pnl', 0) for t in trade_history)
        winning_trades = sum(1 for t in trade_history if t.get('realized_pnl', 0) > 0)
        total_trades = len([t for t in trade_history if t.get('traded', False)])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return f"Total PnL: {total_pnl:.2f} | Win Rate: {win_rate:.1f}%"

    def _figure_to_array(self, fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = iio.imread(buf)
        plt.close(fig)
        return image


def create_advanced_config():
    """Create an advanced configuration with multiple indicators."""
    return [
        {
            'name': 'price',
            'title': 'Price, Trades & Volume Profile',
            'height_ratio': 3,
            'plots': ['price', 'trades', 'volume_profile', 'pivots'],
            'y_label': 'Price'
        },
        {
            'name': 'equity',
            'title': 'Account Equity',
            'height_ratio': 1,
            'plots': ['equity'],
            'y_label': 'Equity'
        },
        {
            'name': 'reward',
            'title': 'Reward',
            'height_ratio': 1,
            'plots': ['reward'],
            'y_label': 'Reward'
        }
    ]
