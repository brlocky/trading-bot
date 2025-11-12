"""
Observation Visualizer - Interactive visualization for multi-input observations.

Visualizes the 8 feature groups from SimpleTradingEnv:
1. Micro Temporal (5): OHLC + Volume
2. Micro Spatial (4): Candle structure
3. Meso Patterns (2): 1h, 4h returns
4. Macro Patterns (1): 24h return
5. Account State (5): Balance, equity, PnL
6. Position Info (7): Position status, leverage, distances
7. VP Bins (n_bins): Volume distribution histogram
8. VP Levels (3): VAH/VAL/POC distances
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict


class ObservationVisualizer:
    """Interactive visualization for trading environment observations."""

    # Feature group metadata
    FEATURE_INFO = {
        'micro_temporal': {
            'title': 'Micro Temporal (OHLC + Volume)',
            'features': ['open', 'high', 'low', 'close', 'volume'],
            'colorscale': 'Viridis',
            'range': [0, 1]
        },
        'micro_spatial': {
            'title': 'Micro Spatial (Candle Structure)',
            'features': ['body_ratio', 'upper_wick', 'lower_wick', 'close_pos'],
            'colorscale': 'Plasma',
            'range': [0, 1]
        },
        'meso_patterns': {
            'title': 'Meso Patterns (1h, 4h Returns)',
            'features': ['return_1h', 'return_4h'],
            'colorscale': 'RdBu',
            'range': [-1, 1]
        },
        'macro_patterns': {
            'title': 'Macro Patterns (24h Return)',
            'features': ['return_24h'],
            'colorscale': 'RdBu',
            'range': [-1, 1]
        },
        'account_state': {
            'title': 'Account State (Balance, PnL)',
            'features': ['equity_growth', 'balance_ratio', 'unrealized_pnl', 'pnl_velocity', 'profit_factor'],
            'colorscale': 'RdYlGn',
            'range': [-1, 1]
        },
        'position_info': {
            'title': 'Position Info (Status, Distances)',
            'features': ['direction', 'leverage', 'unrealized_pnl%', 'sl_dist', 'tp_dist', 'rr_ratio', 'duration'],
            'colorscale': 'Picnic',
            'range': [-1, 1]
        },
        'vp_bins': {
            'title': 'VP Bins (Volume Distribution)',
            'features': None,  # Dynamic based on n_bins
            'colorscale': 'Hot',
            'range': [0, 1]
        },
        'vp_levels': {
            'title': 'VP Levels (VAH/VAL/POC)',
            'features': ['vah_dist', 'val_dist', 'poc_dist'],
            'colorscale': 'RdBu',
            'range': [-1, 1]
        }
    }

    def __init__(self, height_per_group=250):
        """
        Initialize visualizer.

        Args:
            height_per_group: Height in pixels for each feature group subplot
        """
        self.height_per_group = height_per_group

    def visualize(self, obs_dict: Dict[str, np.ndarray], step: int = 0,
                  title: str = None, show_values: bool = True):
        """
        Create interactive heatmap visualization of observation.

        Args:
            obs_dict: Dictionary of observations {group_name: array[lookback, n_features]}
            step: Current step number for title
            title: Custom title (optional)
            show_values: Show hover values

        Returns:
            plotly Figure object
        """
        n_groups = len(obs_dict)

        # Create subplots
        subplot_titles = [
            f"{self.FEATURE_INFO.get(k, {}).get('title', k)} {list(v.shape)}"
            for k, v in obs_dict.items()
        ]

        fig = make_subplots(
            rows=n_groups, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.03,
            row_heights=[1/n_groups] * n_groups
        )

        # Add heatmap for each feature group
        for idx, (name, data) in enumerate(obs_dict.items(), 1):
            info = self.FEATURE_INFO.get(name, {})

            # Get feature names
            if info.get('features'):
                y_labels = info['features']
            elif name == 'vp_bins':
                y_labels = [f'bin_{i}' for i in range(data.shape[1])]
            else:
                y_labels = [f'feat_{i}' for i in range(data.shape[1])]

            # Transpose so features are rows, time is columns
            z_data = data.T

            # Create heatmap
            heatmap = go.Heatmap(
                z=z_data,
                y=y_labels,
                x=list(range(data.shape[0])),  # Timesteps
                colorscale=info.get('colorscale', 'RdBu'),
                zmid=0 if info.get('range', [-1, 1])[0] < 0 else None,
                zmin=info.get('range', [None, None])[0],
                zmax=info.get('range', [None, None])[1],
                colorbar=dict(
                    title=name,
                    x=1.02,
                    y=1 - (idx-0.5)/n_groups,
                    len=1/n_groups * 0.9,
                    thickness=15
                ),
                hovertemplate='Time: %{x}<br>%{y}<br>Value: %{z:.3f}<extra></extra>' if show_values else None
            )

            fig.add_trace(heatmap, row=idx, col=1)

            # Update axes
            fig.update_xaxes(title_text="Timestep →", row=idx, col=1)
            fig.update_yaxes(title_text="", row=idx, col=1)

        # Update layout
        fig.update_layout(
            title=title or f'Observation at Step {step}',
            height=self.height_per_group * n_groups,
            showlegend=False,
            font=dict(size=10)
        )

        return fig

    def visualize_sequence(self, env, n_steps: int = 10, actions=None):
        """
        Visualize a sequence of observations from environment.

        Args:
            env: Trading environment instance
            n_steps: Number of steps to visualize
            actions: List of actions to take (if None, random actions)

        Returns:
            List of plotly figures
        """
        figures = []
        obs, info = env.reset()

        for step in range(n_steps):
            # Create visualization
            fig = self.visualize(obs, step=step, title=f'Step {step} (Episode)')
            figures.append(fig)

            # Take action
            if actions and step < len(actions):
                action = actions[step]
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step([action])

            if done or truncated:
                obs, _ = env.reset()

        return figures

    def compare_observations(self, obs_dict1: Dict[str, np.ndarray],
                             obs_dict2: Dict[str, np.ndarray],
                             titles: tuple = ("Observation 1", "Observation 2")):
        """
        Compare two observations side-by-side.

        Args:
            obs_dict1: First observation dictionary
            obs_dict2: Second observation dictionary
            titles: Tuple of (title1, title2)

        Returns:
            plotly Figure with side-by-side comparison
        """
        n_groups = len(obs_dict1)

        fig = make_subplots(
            rows=n_groups, cols=2,
            subplot_titles=[f"{k} - {titles[0]}" for k in obs_dict1.keys()] +
            [f"{k} - {titles[1]}" for k in obs_dict2.keys()],
            horizontal_spacing=0.05,
            vertical_spacing=0.03
        )

        # Add heatmaps for both observations
        for idx, name in enumerate(obs_dict1.keys(), 1):
            info = self.FEATURE_INFO.get(name, {})

            for col, obs_dict in enumerate([obs_dict1, obs_dict2], 1):
                data = obs_dict[name]

                fig.add_trace(
                    go.Heatmap(
                        z=data.T,
                        colorscale=info.get('colorscale', 'RdBu'),
                        zmid=0 if info.get('range', [-1, 1])[0] < 0 else None,
                        showscale=(col == 2),
                        hovertemplate='Time: %{x}<br>Feat: %{y}<br>Val: %{z:.3f}<extra></extra>'
                    ),
                    row=idx, col=col
                )

        fig.update_layout(
            title=f"Comparison: {titles[0]} vs {titles[1]}",
            height=200 * n_groups,
            showlegend=False
        )

        return fig

    def get_stats_summary(self, obs_dict: Dict[str, np.ndarray]) -> str:
        """
        Generate text summary of observation statistics.

        Args:
            obs_dict: Observation dictionary

        Returns:
            Formatted string with statistics
        """
        summary = "📊 Observation Statistics:\n" + "="*50 + "\n"

        for name, data in obs_dict.items():
            info = self.FEATURE_INFO.get(name, {})
            expected_range = info.get('range', [None, None])

            summary += f"\n{name.upper()} [{data.shape}]:\n"
            summary += f"  Range: [{data.min():.3f}, {data.max():.3f}]"
            if expected_range[0] is not None:
                summary += f" (expected: {expected_range})"
            summary += f"\n  Mean: {data.mean():.3f} | Std: {data.std():.3f}\n"
            summary += f"  NaN: {np.isnan(data).sum()} | Inf: {np.isinf(data).sum()}\n"

        return summary
