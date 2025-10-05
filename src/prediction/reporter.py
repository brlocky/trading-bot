"""
Simple Model Reporter - Comprehensive test reporting with visualizations
"""

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class SimpleModelReporter:
    """
    Comprehensive test reporting system with interactive visualizations
    """

    def __init__(self, model_trainer):
        """
        Args:
            model_trainer: SimpleModelTrainer instance with loaded model
        """
        self.model_trainer = model_trainer

    def generate_full_report(self, signals_df: pd.DataFrame, symbol: str,
                             buy_threshold: float, sell_threshold: float,
                             aggressive_threshold: float) -> None:
        """Generate complete test report with chart and analysis"""
        if signals_df is None or len(signals_df) == 0:
            print("❌ No signals data to report")
            return

        print("\n📈 Creating comprehensive test report...")

        # Get model info
        model_info = self.model_trainer.get_model_info()

        # Create interactive chart
        fig = self._create_interactive_chart(signals_df, symbol, model_info)
        fig.show()

        # Generate detailed analysis
        self._print_detailed_analysis(signals_df, model_info, symbol,
                                      buy_threshold, sell_threshold, aggressive_threshold)

    def _create_interactive_chart(self, signals_df: pd.DataFrame, symbol: str, model_info: dict):
        """Create the 3-panel interactive Plotly chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{symbol} - {model_info["model_type"]} Predictions',
                'Confidence Score',
                'Probability Breakdown'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=signals_df['datetime'],
                open=signals_df['open'],
                high=signals_df['high'],
                low=signals_df['low'],
                close=signals_df['close'],
                name='Price',
                showlegend=True
            ),
            row=1, col=1
        )

        # BUY signals
        buy_signals = signals_df[signals_df['action'] == 'buy']
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['datetime'],
                    y=buy_signals['close'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='lime'),
                    name='BUY Prediction',
                    text=buy_signals['reasoning'],
                    hovertemplate='<b>BUY</b><br>Price: $%{y:.4f}<br>Confidence: %{customdata:.1%}<br>%{text}<extra></extra>',
                    customdata=buy_signals['confidence']
                ),
                row=1, col=1
            )

        # SELL signals
        sell_signals = signals_df[signals_df['action'] == 'sell']
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['datetime'],
                    y=sell_signals['close'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='SELL Prediction',
                    text=sell_signals['reasoning'],
                    hovertemplate='<b>SELL</b><br>Price: $%{y:.4f}<br>Confidence: %{customdata:.1%}<br>%{text}<extra></extra>',
                    customdata=sell_signals['confidence']
                ),
                row=1, col=1
            )

        # HOLD signals (smaller, less prominent)
        hold_signals = signals_df[signals_df['action'] == 'hold']
        if not hold_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=hold_signals['datetime'],
                    y=hold_signals['close'],
                    mode='markers',
                    marker=dict(symbol='circle', size=6, color='gray', opacity=0.5),
                    name='HOLD Prediction',
                    text=hold_signals['reasoning'],
                    hovertemplate='<b>HOLD</b><br>Price: $%{y:.4f}<br>Confidence: %{customdata:.1%}<br>%{text}<extra></extra>',
                    customdata=hold_signals['confidence']
                ),
                row=1, col=1
            )

        # Confidence line
        confidence_colors = ['lime' if action == 'buy' else 'red' if action == 'sell' else 'gray'
                             for action in signals_df['action']]

        fig.add_trace(
            go.Scatter(
                x=signals_df['datetime'],
                y=signals_df['confidence'],
                mode='markers+lines',
                name='Model Confidence',
                line=dict(color='white', width=1),
                marker=dict(color=confidence_colors, size=8),
                hovertemplate='Confidence: %{y:.1%}<br>Prediction: %{customdata}<extra></extra>',
                customdata=signals_df['action']
            ),
            row=2, col=1
        )

        # Probability breakdown (stacked area chart)
        fig.add_trace(
            go.Scatter(
                x=signals_df['datetime'],
                y=signals_df['buy_prob'],
                fill='tonexty',
                mode='none',
                name='Buy Probability',
                fillcolor='rgba(0, 255, 0, 0.3)',
                hovertemplate='Buy Prob: %{y:.1%}<extra></extra>'
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=signals_df['datetime'],
                y=signals_df['sell_prob'],
                fill='tonexty',
                mode='none',
                name='Sell Probability',
                fillcolor='rgba(255, 0, 0, 0.3)',
                hovertemplate='Sell Prob: %{y:.1%}<extra></extra>'
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'🤖 {model_info["model_type"]} AGGRESSIVE Predictions on {symbol}',
            height=900,
            template='plotly_dark',
            showlegend=True,
            # Enable mouse zoom and disable range selector
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date'
            ),
            xaxis2=dict(
                rangeslider=dict(visible=False),
                type='date'
            ),
            xaxis3=dict(
                rangeslider=dict(visible=False),
                type='date'
            ),
            # Enable drag mode for zooming
            dragmode='zoom',
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1, tickformat='.0%')
        fig.update_yaxes(title_text="Probability", row=3, col=1, tickformat='.0%')
        fig.update_xaxes(title_text="Time", row=3, col=1)

        return fig

    def _print_detailed_analysis(self, signals_df: pd.DataFrame, model_info: dict, symbol: str,
                                 buy_threshold: float, sell_threshold: float, aggressive_threshold: float):
        """Print comprehensive analysis report"""

        print("\n🔍 DETAILED ANALYSIS:")
        print("=" * 50)

        # Model and configuration info
        print(f"🤖 Model: {model_info['model_type']} (Accuracy: {model_info['accuracy']:.1%})")
        print(f"📊 Features: {model_info['features']} | Classes: {model_info['classes']}")
        print(f"🎯 Symbol: {symbol} | Predictions: {len(signals_df)}")
        print(f"⚙️  Thresholds: BUY≥{buy_threshold:.0%}, SELL≥{sell_threshold:.0%}, Aggressive≥{aggressive_threshold:.0%}")

        # Feature consistency check
        feature_columns = self.model_trainer.model_data.get('feature_columns', [])
        print(f"\n🔧 Feature Consistency Check:")
        print(f"   Training features: {len(feature_columns)}")
        if len(feature_columns) > 0:
            print(f"   Key features: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")

        # Action distribution
        action_counts = signals_df['action'].value_counts()
        total_signals = len(signals_df)

        print(f"\n📊 Signal Distribution:")
        for action in ['buy', 'sell', 'hold']:
            count = action_counts.get(action, 0)
            pct = count / total_signals * 100
            print(f"   {action.upper()}: {count:,} ({pct:.1f}%)")

        # Trade signals (non-hold)
        trade_signals = signals_df[signals_df['action'] != 'hold']
        trade_pct = len(trade_signals) / total_signals * 100
        print(f"\n🚀 Trade Opportunities: {len(trade_signals):,} signals ({trade_pct:.1f}% of time)")

        # High-confidence predictions
        high_conf = signals_df[signals_df['confidence'] > 0.6]
        if not high_conf.empty:
            print(f"\n💪 High Confidence Predictions (>60%): {len(high_conf)}")
            for _, pred in high_conf.head(5).iterrows():
                print(f"   {pred['datetime'].strftime('%Y-%m-%d %H:%M')}: " +
                      f"{pred['action'].upper()} @ ${pred['close']:.4f} ({pred['confidence']:.1%})")
            if len(high_conf) > 5:
                print(f"   ... and {len(high_conf) - 5} more")

        # Probability analysis
        avg_buy_prob = signals_df['buy_prob'].mean()
        avg_sell_prob = signals_df['sell_prob'].mean()
        avg_hold_prob = signals_df['hold_prob'].mean()

        print(f"\n🎯 Average Probabilities:")
        print(f"   BUY: {avg_buy_prob:.1%} | SELL: {avg_sell_prob:.1%} | HOLD: {avg_hold_prob:.1%}")

        # Performance metrics
        avg_confidence = signals_df['confidence'].mean()
        max_confidence = signals_df['confidence'].max()
        min_confidence = signals_df['confidence'].min()

        print(f"\n📈 Confidence Metrics:")
        print(f"   Average: {avg_confidence:.1%} | Max: {max_confidence:.1%} | Min: {min_confidence:.1%}")

        print(f"\n✅ TESTING COMPLETE!")
        print("=" * 50)
