"""
Model Testing Report

Uses TradingEnvironment for all trading simulation - no reinventing the wheel!
Shows price data, buy/sell signals, and PnL tracking from Environment.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from environments.trading_environment import TradingEnvironment
from core.normalization_config import get_default_environment_config


def simple_trading_chart(features_df, predictions, symbol='SYMBOL',
                         buy_threshold=0.1, sell_threshold=-0.1,
                         normalizer=None, feature_config=None):
    """
    ✅ FIXED: Uses TradingEnvironment for ALL trading simulation (no reinventing wheel)

    Args:
        features_df: DataFrame with feature data
        predictions: Array of model predictions (-1 to 1)
        symbol: Symbol name for title
        buy_threshold: Threshold for BUY signals (default: 0.1)
        sell_threshold: Threshold for SELL signals (default: -0.1)
        normalizer: Pre-fitted normalizer (optional)
        feature_config: Feature configuration (optional)

    Returns:
        Dict with trading metrics from Environment
    """

    print("✅ Using TradingEnvironment for ALL trading simulation")
    print(f"   • BUY threshold: {buy_threshold}")
    print(f"   • SELL threshold: {sell_threshold}")

    # ✅ Create Environment to handle ALL trading logic
    env_config = get_default_environment_config()

    # Create dummy feature config if not provided
    if feature_config is None:
        feature_config = {col: 'minmax' for col in features_df.columns
                          if col not in ['time', 'timestamp']}

    # Create Environment (same as RL predictor)
    env = TradingEnvironment(
        df=features_df,
        feature_config=feature_config,
        fit_normalizer=False if normalizer else True,
        **env_config
    )

    obs, _ = env.reset()
    initial_portfolio = env.balance

    # ✅ Simulate trading using Environment (not manual calculations)
    actions = []
    portfolio_values = []
    positions = []
    rewards = []
    step_pnls = []
    trade_log = []

    print(f"🎯 Simulating {len(features_df) - env.window_size} steps using Environment...")

    for i in range(env.window_size, len(features_df)):
        # Map predictions to actions using thresholds
        pred_value = predictions[i] if i < len(predictions) else 0.0

        if pd.isna(pred_value):
            action = np.array([0.0])  # HOLD for NaN predictions
        elif pred_value > buy_threshold:
            action = np.array([pred_value])  # BUY signal
        elif pred_value < sell_threshold:
            action = np.array([pred_value])  # SELL signal
        else:
            action = np.array([0.0])  # HOLD

        # ✅ Environment handles ALL trading logic
        obs, reward, done, _, info = env.step(action)

        # Collect data FROM environment
        actions.append(float(action[0]))
        portfolio_values.append(env.balance)
        positions.append(env.position)
        rewards.append(reward)
        step_pnls.append(info.get('step_pnl', 0.0))

        # Track individual trades
        if info.get('trade_occurred', False):
            trade_log.append({
                'step': i,
                'price': info.get('price', features_df.iloc[i]['close']),
                'action': float(action[0]),
                'step_pnl': info.get('step_pnl', 0.0)
            })

        if done:
            break

    # ✅ Create results DataFrame using Environment data
    result_df = features_df.iloc[env.window_size:env.window_size+len(actions)].copy()
    result_df['prediction'] = actions
    result_df['portfolio_value'] = portfolio_values
    result_df['position'] = positions
    result_df['step_pnl'] = step_pnls

    # Map actions to signals for visualization
    result_df['signal'] = 'HOLD'
    result_df['action'] = 0

    buy_mask = np.array(actions) > buy_threshold
    sell_mask = np.array(actions) < sell_threshold

    result_df.loc[buy_mask, 'signal'] = 'BUY'
    result_df.loc[buy_mask, 'action'] = 1
    result_df.loc[sell_mask, 'signal'] = 'SELL'
    result_df.loc[sell_mask, 'action'] = -1

    # Calculate metrics
    result_df['pnl_pct'] = ((np.array(portfolio_values) / initial_portfolio) - 1) * 100
    result_df['buyhold_pct'] = (result_df['close'] / result_df['close'].iloc[0] - 1) * 100

    print(f"🔧 Environment Signal Mapping:")
    print(f"   • BUY signals (>{buy_threshold}): {buy_mask.sum():,}")
    print(f"   • SELL signals (<{sell_threshold}): {sell_mask.sum():,}")
    print(f"   • HOLD signals: {(~buy_mask & ~sell_mask).sum():,}")
    print(f"   • Total trades: {len(trade_log):,}")

    # ✅ Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f'{symbol} - Price & Signals (Environment Simulation)',
            'PnL Comparison (Environment vs Buy&Hold)'
        ],
        row_heights=[0.6, 0.4]
    )

    # Price chart
    fig.add_trace(
        go.Scatter(x=result_df['time'], y=result_df['close'],
                   mode='lines', name='Price', line=dict(color='black')),
        row=1, col=1
    )

    # Buy signals
    buys = result_df[result_df['action'] == 1]
    if len(buys) > 0:
        fig.add_trace(
            go.Scatter(
                x=buys['time'], y=buys['close'], mode='markers',
                name=f'BUY ({len(buys)})',
                marker=dict(symbol='triangle-up', size=8, color='green')
            ),
            row=1, col=1
        )

    # Sell signals
    sells = result_df[result_df['action'] == -1]
    if len(sells) > 0:
        fig.add_trace(
            go.Scatter(
                x=sells['time'], y=sells['close'], mode='markers',
                name=f'SELL ({len(sells)})',
                marker=dict(symbol='triangle-down', size=8, color='red')
            ),
            row=1, col=1
        )

    # PnL chart
    fig.add_trace(
        go.Scatter(x=result_df['time'], y=result_df['pnl_pct'],
                   mode='lines', name='Environment Strategy', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=result_df['time'], y=result_df['buyhold_pct'],
                   mode='lines', name='Buy & Hold', line=dict(color='orange')),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title=f'Trading Performance: {symbol} (Environment Simulation)',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return %", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    # ✅ Calculate final stats using Environment data
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_portfolio
    final_return = ((final_portfolio / initial_portfolio) - 1) * 100
    buyhold_return = result_df['buyhold_pct'].iloc[-1] if len(result_df) > 0 else 0
    max_drawdown = ((np.array(portfolio_values) / np.maximum.accumulate(portfolio_values)) - 1).min() * 100

    # Win rate from trade log
    profitable_trades = len([t for t in trade_log if t['step_pnl'] > 0])
    win_rate = (profitable_trades / len(trade_log) * 100) if trade_log else 0

    stats = {
        'symbol': symbol,
        'final_return_pct': final_return,
        'buyhold_return_pct': buyhold_return,
        'outperformance_pct': final_return - buyhold_return,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'total_trades': len(trade_log),
        'final_value': final_portfolio,
        'initial_value': initial_portfolio,
        'trade_log': trade_log
    }

    # Show chart
    fig.show()

    # Print stats
    print(f"\n📊 {symbol} Environment Trading Results:")
    print(f"   Strategy Return: {final_return:+.2f}%")
    print(f"   Buy & Hold:      {buyhold_return:+.2f}%")
    print(f"   Outperformance:  {final_return - buyhold_return:+.2f}%")
    print(f"   Max Drawdown:    {max_drawdown:.2f}%")
    print(f"   Win Rate:        {win_rate:.1f}%")
    print(f"   Total Trades:    {len(trade_log):,}")
    print(f"   Final Value:     ${final_portfolio:,.2f}")
    print(f"   Initial Value:   ${initial_portfolio:,.2f}")

    return stats
    df['pnl_pct'] = (df['portfolio_value'] / initial_cash - 1) * 100
    df['buyhold_pct'] = (df['close'] / df['close'].iloc[0] - 1) * 100

    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f'{symbol} - Price & Signals',
            'PnL Comparison'
        ],
        row_heights=[0.6, 0.4]
    )

    # Price chart
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['close'], mode='lines', name='Price', line=dict(color='black')),
        row=1, col=1
    )

    # Buy signals
    buys = df[df['action'] == 1]
    if len(buys) > 0:
        fig.add_trace(
            go.Scatter(
                x=buys['time'], y=buys['close'], mode='markers',
                name=f'BUY ({len(buys)})', marker=dict(symbol='triangle-up', size=8, color='green')
            ),
            row=1, col=1
        )

    # Sell signals
    sells = df[df['action'] == -1]
    if len(sells) > 0:
        fig.add_trace(
            go.Scatter(
                x=sells['time'], y=sells['close'], mode='markers',
                name=f'SELL ({len(sells)})', marker=dict(symbol='triangle-down', size=8, color='red')
            ),
            row=1, col=1
        )

    # PnL chart
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['pnl_pct'], mode='lines', name='Strategy', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['buyhold_pct'], mode='lines', name='Buy & Hold', line=dict(color='orange')),
        row=2, col=1
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Layout
    fig.update_layout(
        title=f'Trading Performance: {symbol}',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return %", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    # Calculate stats
    final_return = df['pnl_pct'].iloc[-1]
    buyhold_return = df['buyhold_pct'].iloc[-1]
    max_drawdown = (df['portfolio_value'] / df['portfolio_value'].expanding().max() - 1).min() * 100

    # Win rate calculation
    trade_returns = []
    for i in range(len(df) - 1):
        if df.iloc[i]['action'] != 0:
            next_return = (df.iloc[i+1]['close'] / df.iloc[i]['close'] - 1) * 100
            if df.iloc[i]['action'] == 1:
                trade_returns.append(next_return)
            else:
                trade_returns.append(-next_return)

    win_rate = (np.array(trade_returns) > 0).mean() * 100 if trade_returns else 0

    stats = {
        'symbol': symbol,
        'final_return_pct': final_return,
        'buyhold_return_pct': buyhold_return,
        'outperformance_pct': final_return - buyhold_return,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'total_trades': len(buys) + len(sells),
        'final_value': df['portfolio_value'].iloc[-1]
    }

    # Show chart
    fig.show()

    # Print stats
    print(f"\n📊 {symbol} Trading Results:")
    print(f"   Strategy Return: {final_return:+.2f}%")
    print(f"   Buy & Hold:      {buyhold_return:+.2f}%")
    print(f"   Outperformance:  {final_return - buyhold_return:+.2f}%")
    print(f"   Max Drawdown:    {max_drawdown:.2f}%")
    print(f"   Win Rate:        {win_rate:.1f}%")
    print(f"   Total Trades:    {len(buys) + len(sells)}")
    print(f"   Final Value:     ${df['portfolio_value'].iloc[-1]:,.2f}")

    return stats


def show_trading_results(features_df, predictions, symbol='CRYPTO',
                         buy_threshold=0.1, sell_threshold=-0.1,
                         normalizer=None, feature_config=None):
    """
    ✅ ENVIRONMENT-BASED: Uses TradingEnvironment for all trading simulation

    Usage:
        show_trading_results(features_df, rl_actions, 'BTCUSDT',
                            buy_threshold=0.1, sell_threshold=-0.1,
                            normalizer=rl_predictor.normalizer,
                            feature_config=rl_predictor.feature_config)
    """
    return simple_trading_chart(features_df, predictions, symbol,
                                buy_threshold=buy_threshold,
                                sell_threshold=sell_threshold,
                                normalizer=normalizer,
                                feature_config=feature_config)
