"""
VectorBT Backtesting Engine

Replaces the manual candle-by-candle simulation with VectorBT's
vectorized backtesting for high-performance strategy evaluation.
"""

import pandas as pd
import vectorbt as vbt
from typing import Dict, Optional, Any
from pathlib import Path


class VectorBTBacktester:
    """
    VectorBT-powered backtesting engine for ML trading strategies.

    This class provides:
    - Vectorized signal generation from ML predictions
    - High-performance backtesting with realistic commissions/slippage
    - Comprehensive performance metrics
    - Integration with TradeMemoryManager
    - Multi-timeframe support
    """

    def __init__(
        self,
        trainer,
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize VectorBT backtester.

        Args:
            trainer: SimpleModelTrainer instance with trained model
            initial_cash: Starting capital for backtesting
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
        """
        self.trainer = trainer
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.portfolio = None
        self.signals_df = None

    def generate_signals(
        self,
        data: pd.DataFrame,
        level_files: Dict[str, str],
        buy_threshold: float = 0.10,
        sell_threshold: float = 0.10,
    ) -> pd.DataFrame:
        """
        Generate trading signals from ML model predictions.

        Args:
            data: DataFrame with OHLCV data (must have datetime index)
            level_files: Dict of timeframe -> file path for levels
            buy_threshold: Minimum probability for BUY signal
            sell_threshold: Minimum probability for SELL signal

        Returns:
            DataFrame with columns: datetime, close, buy_prob, sell_prob, hold_prob,
                                   signal, confidence, entries, exits
        """
        if not self.trainer.is_trained:
            raise ValueError("Model is not trained. Train the model first.")

        print(f"🎯 Generating signals for {len(data)} candles...")

        # Ensure datetime is in index
        if 'datetime' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('datetime')

        # Load levels
        from src.prediction.predictor import SimpleModelPredictor
        import time
        
        predictor = SimpleModelPredictor(self.trainer)

        if level_files:
            print(f"📊 Loading levels from: {list(level_files.keys())}")
            print("   ⏳ This may take 30-60 seconds (loading historical data)...")
            start_time = time.time()
            
            success = predictor.trader.update_levels(level_files, force_update=True)
            
            elapsed = time.time() - start_time
            if success:
                total_levels = sum(
                    len(levels) for levels in predictor.trader.current_levels.values()
                )
                print(f"✅ Loaded {total_levels} support/resistance levels in {elapsed:.1f}s")
            else:
                print("⚠️  Failed to load levels")

        # Get model components
        model_data = self.trainer.model_data
        trained_model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_columns = model_data['feature_columns']

        # Generate predictions for all candles
        predictions = []
        
        # Get memory features once (they don't change per candle)
        memory_features = {}
        if self.trainer.enable_memory_features:
            recent_perf = self.trainer.trade_memory.get_recent_performance()
            bounce_perf = self.trainer.trade_memory.get_bounce_performance()
            consecutive = self.trainer.trade_memory.get_consecutive_performance()

            memory_features = {
                'memory_win_rate': recent_perf['win_rate'],
                'memory_avg_pnl': recent_perf['avg_pnl'],
                'memory_total_trades': recent_perf['total_trades'],
                'bounce_win_rate': bounce_perf['bounce_win_rate'],
                'bounce_avg_pnl': bounce_perf['bounce_avg_pnl'],
                'bounce_trade_count': bounce_perf['bounce_trades'],
                'consecutive_wins': max(0, consecutive),
                'consecutive_losses': max(0, -consecutive),
                'market_volatility_regime': 0.5,
                'trend_strength': 0.0,
            }

        print(f"   Processing {len(data)} candles...")
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            try:
                current_price = float(row['close'])
                current_volume = float(row['volume'])

                # Create features
                features = predictor.trader.feature_engineer.create_level_features(
                    current_price, current_volume, predictor.trader.current_levels
                )

                # Add memory features (already computed once)
                if memory_features:
                    features.update(memory_features)

                # Convert to DataFrame and align with training features
                feature_df = pd.DataFrame([features])

                # Ensure exact feature match with training
                for col in feature_columns:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0
                feature_df = feature_df[feature_columns]

                # Make prediction
                probabilities = trained_model.predict_proba(feature_df)[0]
                classes = label_encoder.classes_
                prob_dict = {classes[j]: probabilities[j] for j in range(len(classes))}

                buy_prob = prob_dict.get('buy', 0)
                sell_prob = prob_dict.get('sell', 0)
                hold_prob = prob_dict.get('hold', 0)

                # Apply thresholds
                if buy_prob > buy_threshold and buy_prob > sell_prob:
                    signal = 'BUY'
                    confidence = buy_prob
                elif sell_prob > sell_threshold and sell_prob > buy_prob:
                    signal = 'SELL'
                    confidence = sell_prob
                else:
                    signal = 'HOLD'
                    confidence = hold_prob

                predictions.append({
                    'datetime': timestamp,
                    'close': current_price,
                    'buy_prob': buy_prob,
                    'sell_prob': sell_prob,
                    'hold_prob': hold_prob,
                    'signal': signal,
                    'confidence': confidence,
                })

                if (i + 1) % 10 == 0 or (i + 1) == len(data):
                    print(f"   Processed {i+1}/{len(data)} candles...")

            except Exception as e:
                print(f"❌ Error processing candle {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Convert to DataFrame
        signals_df = pd.DataFrame(predictions)
        signals_df = signals_df.set_index('datetime')

        # Create entry/exit signals for VectorBT
        # Entry: BUY signal (1), Exit: SELL signal (1), else 0
        signals_df['entries'] = (signals_df['signal'] == 'BUY').astype(int)
        signals_df['exits'] = (signals_df['signal'] == 'SELL').astype(int)

        print(f"✅ Generated {len(signals_df)} signals")
        print(f"   BUY: {(signals_df['signal'] == 'BUY').sum()}")
        print(f"   SELL: {(signals_df['signal'] == 'SELL').sum()}")
        print(f"   HOLD: {(signals_df['signal'] == 'HOLD').sum()}")

        self.signals_df = signals_df
        return signals_df

    def run_backtest(
        self,
        signals_df: Optional[pd.DataFrame] = None,
        freq: str = '15T',  # 15-minute frequency
    ) -> vbt.Portfolio:
        """
        Run VectorBT backtest on generated signals.

        Args:
            signals_df: DataFrame with entries/exits (uses self.signals_df if None)
            freq: Frequency string for VectorBT (e.g., '15T', '1H', '1D')

        Returns:
            VectorBT Portfolio object with results
        """
        if signals_df is None:
            if self.signals_df is None:
                raise ValueError("No signals available. Run generate_signals() first.")
            signals_df = self.signals_df

        print("\n🚀 Running VectorBT backtest...")
        print(f"   Initial Capital: ${self.initial_cash:,.2f}")
        print(f"   Commission: {self.commission:.2%}")
        print(f"   Slippage: {self.slippage:.2%}")

        # Create Portfolio using from_signals
        portfolio = vbt.Portfolio.from_signals(
            close=signals_df['close'],
            entries=signals_df['entries'],
            exits=signals_df['exits'],
            init_cash=self.initial_cash,
            fees=self.commission,
            slippage=self.slippage,
            freq=freq,
        )

        self.portfolio = portfolio
        print("✅ Backtest complete!")

        return portfolio

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        stats = self.portfolio.stats()

        # Convert to dictionary for easier access
        stats_dict = {
            'total_return': stats['Total Return [%]'],
            'annualized_return': stats.get('Annual Return [%]', 0),
            'sharpe_ratio': stats.get('Sharpe Ratio', 0),
            'max_drawdown': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'total_trades': stats['Total Trades'],
            'profit_factor': stats.get('Profit Factor', 0),
            'final_value': stats['End Value'],
        }

        return stats_dict

    def print_performance_summary(self):
        """Print a formatted performance summary."""
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        print("\n" + "=" * 60)
        print("📊 VECTORBT BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)

        stats = self.get_performance_stats()

        print("\n💰 Returns:")
        print(f"   Total Return:      {stats['total_return']:>10.2f}%")
        print(f"   Annualized Return: {stats['annualized_return']:>10.2f}%")
        print(f"   Final Value:       ${stats['final_value']:>10,.2f}")

        print("\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:      {stats['sharpe_ratio']:>10.2f}")
        print(f"   Max Drawdown:      {stats['max_drawdown']:>10.2f}%")

        print("\n🎯 Trading Performance:")
        print(f"   Total Trades:      {stats['total_trades']:>10}")
        print(f"   Win Rate:          {stats['win_rate']:>10.2f}%")
        print(f"   Profit Factor:     {stats['profit_factor']:>10.2f}")

        print("\n" + "=" * 60)

    def plot_results(self, use_widgets=False):
        """
        Generate interactive VectorBT plots.

        This creates:
        - Equity curve
        - Drawdown chart
        - Trade markers on price chart
        
        Args:
            use_widgets: If True, uses FigureWidget (requires anywidget).
                        If False, uses static Figure (no dependencies).
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        print("\n📊 Generating interactive plots...")

        try:
            # Try with widgets first
            if use_widgets:
                fig = self.portfolio.plot()
            else:
                # Use static plots (no widgets required)
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Get portfolio value over time
                portfolio_value = self.portfolio.value()
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Portfolio Value', 'Drawdown %')
                )
                
                # Add portfolio value trace
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value.values,
                        name='Portfolio Value',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Add initial cash line
                fig.add_hline(
                    y=self.initial_cash,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Initial Cash",
                    row=1, col=1
                )
                
                # Calculate and add drawdown
                drawdown = self.portfolio.drawdown()
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values * 100,  # Convert to percentage
                        name='Drawdown %',
                        fill='tozeroy',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title='VectorBT Backtest Results',
                    height=800,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Value ($)", row=1, col=1)
                fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            fig.show()
            print("✅ Plots displayed!")
            
        except ImportError as e:
            if "anywidget" in str(e):
                print("\n⚠️  anywidget not installed!")
                print("   Solution 1: Restart the kernel (Kernel → Restart)")
                print("   Solution 2: Run: backtester.plot_results(use_widgets=False)")
                print("   Solution 3: Install anywidget and restart kernel")
            else:
                raise

    def get_trade_analysis(self) -> pd.DataFrame:
        """
        Get detailed trade-by-trade analysis.

        Returns:
            DataFrame with individual trade details
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        trades = self.portfolio.trades.records_readable
        return trades

    def export_results(self, output_dir: str = "backtest_results"):
        """
        Export backtest results to files.

        Args:
            output_dir: Directory to save results
        """
        if self.portfolio is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export signals
        signals_file = output_path / "signals.csv"
        self.signals_df.to_csv(signals_file)
        print(f"✅ Saved signals to: {signals_file}")

        # Export trades
        trades_file = output_path / "trades.csv"
        trades = self.get_trade_analysis()
        trades.to_csv(trades_file, index=False)
        print(f"✅ Saved trades to: {trades_file}")

        # Export performance stats
        stats_file = output_path / "performance_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(str(self.portfolio.stats()))
        print(f"✅ Saved performance stats to: {stats_file}")

        # Export equity curve
        equity_file = output_path / "equity_curve.csv"
        equity = self.portfolio.value()
        equity.to_csv(equity_file)
        print(f"✅ Saved equity curve to: {equity_file}")

        print(f"\n📁 All results exported to: {output_path.absolute()}")
