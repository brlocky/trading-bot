"""
Simple Model Testing Report for RL Trading Bot - KISS Implementation
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_simple_charts(results_df):
    """Create simple 5-panel trading visualization"""

    # Create 5 subplots
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Price & Signals', 'Position Size (Input)', 'Position Shares (Held)', 'Balance', 'Step P&L']
    )

    # Panel 1: Price and signals
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df['price'],
            mode='lines',
            name='Price',
            line=dict(color='rgb(0, 100, 255)', width=4)
        ),
        row=1, col=1
    )

    # Add buy/sell signals
    buy_signals = results_df[results_df['signal'] == 1]
    sell_signals = results_df[results_df['signal'] == -1]

    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='Buy',
                marker=dict(color='rgb(0, 255, 0)', size=16, symbol='triangle-up', line=dict(color='rgb(0, 128, 0)', width=3))
            ),
            row=1, col=1
        )

    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='Sell',
                marker=dict(color='rgb(255, 0, 0)', size=16, symbol='triangle-down', line=dict(color='rgb(128, 0, 0)', width=3))
            ),
            row=1, col=1
        )

    # Panel 2: Position Size (Input)
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df['position_size'],
            mode='lines',
            name='Position Size (Input)',
            line=dict(color='rgb(153, 0, 255)', width=4),
            fill='tozeroy',
            fillcolor='rgba(153, 0, 255, 0.5)'
        ),
        row=2, col=1
    )

    # Panel 3: Position Shares (Held)
    # Separate positive (long) and negative (short) positions
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df['position_shares'],
            mode='lines',
            name='Position Shares',
            line=dict(color='rgb(100, 100, 100)', width=2),
            fill='tozeroy',
            fillcolor='rgba(100, 100, 100, 0.3)'
        ),
        row=3, col=1
    )

    # Add green fill for long positions (positive)
    long_shares = results_df['position_shares'].copy()
    long_shares[long_shares < 0] = 0
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=long_shares,
            mode='lines',
            name='Long',
            line=dict(color='rgb(0, 255, 0)', width=0),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.5)',
            showlegend=False
        ),
        row=3, col=1
    )

    # Add red fill for short positions (negative)
    short_shares = results_df['position_shares'].copy()
    short_shares[short_shares > 0] = 0
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=short_shares,
            mode='lines',
            name='Short',
            line=dict(color='rgb(255, 0, 0)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.5)',
            showlegend=False
        ),
        row=3, col=1
    )

    # Panel 4: Balance
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df['cash'] + results_df['unrealized_pnl'],
            mode='lines',
            name='Balance',
            line=dict(color='rgb(0, 200, 0)', width=4)
        ),
        row=4, col=1
    )

    # Panel 5: Step P&L as line chart with conditional coloring
    # Separate positive and negative P&L for better visualization
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df['step_pnl'],
            mode='lines',
            name='Step P&L',
            line=dict(color='rgb(100, 100, 100)', width=2),
            fill='tozeroy',
            fillcolor='rgba(100, 100, 100, 0.3)'
        ),
        row=5, col=1
    )

    # Add green fill for positive P&L
    positive_pnl = results_df['step_pnl'].copy()
    positive_pnl[positive_pnl < 0] = 0
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=positive_pnl,
            mode='lines',
            name='Profit',
            line=dict(color='rgb(0, 255, 0)', width=0),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.5)',
            showlegend=False
        ),
        row=5, col=1
    )

    # Add red fill for negative P&L
    negative_pnl = results_df['step_pnl'].copy()
    negative_pnl[negative_pnl > 0] = 0
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=negative_pnl,
            mode='lines',
            name='Loss',
            line=dict(color='rgb(255, 0, 0)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.5)',
            showlegend=False
        ),
        row=5, col=1
    )

    # Update layout
    fig.update_layout(title='Trading Model Results', height=1100)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Position Size (Input)", row=2, col=1)
    fig.update_yaxes(title_text="Position Shares (Held)", row=3, col=1)
    fig.update_yaxes(title_text="Balance", row=4, col=1)
    fig.update_yaxes(title_text="Step P&L", row=5, col=1)

    return fig


def print_simple_metrics(results_df):
    """Print simple performance metrics"""

    total_pnl = results_df['step_pnl'].sum()
    final_balance = results_df['cash'].iloc[-1]
    total_trades = results_df['traded'].sum()

    print("\n📊 SIMPLE PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"💰 Total P&L: ${total_pnl:,.2f}")
    print(f"💼 Final Balance: ${final_balance:,.2f}")
    print(f"🔄 Total Trades: {total_trades}")


def quick_test_model(model_dir, symbol='BTCUSDT', test_candles=500):
    """Simple model testing function - KISS implementation"""

    print(f"🔍 Testing model: {model_dir}")
    print(f"📊 Symbol: {symbol}, Candles: {test_candles}")

    try:
        # Import required modules
        from prediction.rl_predictor import RLPredictor
        from training.data_loader import DataLoader

        # Load data
        print("📊 Loading data...")
        loader = DataLoader()
        dfs = loader.load_data(symbol)
        test_data = dfs['15m'].tail(test_candles)

        # Create predictor and generate predictions
        print("🤖 Loading model and generating predictions...")
        predictor = RLPredictor(model_dir=model_dir)
        results_df = predictor.generate_predictions(test_data)

        print(f"✅ Test completed! Shape: {results_df.shape}")
        print(f"📊 Columns: {list(results_df.columns)}")

        # Show simple metrics
        print_simple_metrics(results_df)

        # Create and show chart
        fig = create_simple_charts(results_df)
        fig.show()

        return results_df

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
