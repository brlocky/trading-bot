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

    Uses AutonomousTrader as the single source of truth for feature calculation,
    ensuring consistency between training, backtesting, and live trading.

    This class provides:
    - Vectorized signal generation from ML predictions
    - High-performance backtesting with realistic commissions/slippage
    - Comprehensive performance metrics
    - Multi-timeframe support with intelligent caching
    - Automatic day change detection for level features
    """

    def __init__(
        self,
        trainer,
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
        middleware_config: Optional[Dict] = None,
    ):
        """
        Initialize VectorBT backtester.

        Args:
            trainer: SimpleModelTrainer instance with trained model
            initial_cash: Starting capital for backtesting
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            middleware_config: Optional custom middleware configuration per timeframe
                             (passed to trainer's AutonomousTrader)
        """
        self.trainer = trainer
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.portfolio = None
        self.signals_df: Optional[pd.DataFrame] = None

        # No more FeatureManager! Use trainer's AutonomousTrader instead
        # All feature calculation happens in AutonomousTrader (single source of truth)

    def _load_levels_from_df(self, data_dfs: Dict[str, pd.DataFrame], max_date=None) -> Dict:
        """
        Load levels from DataFrames using trainer's AutonomousTrader configured extractor.
        Uses the level_extraction_config from trainer's AutonomousTrader initialization.

        Args:
            data_dfs: Dict of timeframe -> DataFrame with levels
            max_date: Optional datetime to filter data (only use candles <= max_date)
                     Critical for backtesting to prevent data leakage!
        """
        # Use trainer's AutonomousTrader extractor (already configured with level_extraction_config)
        try:
            all_levels = self.trainer.autonomous_trader.level_extractor.extract_levels_from_dataframes(
                data_dfs,
                max_date=max_date
            )
            for tf, levels in all_levels.items():
                print(f"   ✅ Loaded {len(levels)} levels from {tf}")
            return all_levels
        except Exception as e:
            print(f"   ⚠️  Error loading levels: {e}")
            # Return empty dict for each timeframe
            return {tf: [] for tf in data_dfs.keys()}

    def generate_signals(
        self,
        data: pd.DataFrame,
        data_dfs: Dict[str, pd.DataFrame],
        buy_threshold: float = 0.10,
        sell_threshold: float = 0.10,
        timeframe: str = '15m',
    ) -> pd.DataFrame:
        """
        Generate trading signals PROGRESSIVELY - one candle at a time.

        For EACH candle:
        1. Extract levels from all timeframes (only data up to current candle timestamp)
        2. Calculate features for that candle
        3. Make prediction
        4. Move to next candle

        This simulates REAL TRADING with no data leakage.

        Args:
            data: DataFrame with OHLCV data (must have datetime index)
            data_dfs: Dict of timeframe -> DataFrame with levels
            buy_threshold: Minimum probability for BUY signal
            sell_threshold: Minimum probability for SELL signal
            timeframe: Timeframe for feature calculation (e.g., '15m', '1h', 'D')

        Returns:
            DataFrame with columns: datetime, close, buy_prob, sell_prob, hold_prob,
                                   signal, confidence, entries, exits
        """
        if not self.trainer.is_trained:
            raise ValueError("Model is not trained. Train the model first.")

        print(f"🎯 PROGRESSIVE BACKTESTING: Processing {len(data)} candles...")
        print("⚠️  This will take longer as we extract levels for EACH candle")
        print("   (simulates real trading - no data leakage!)\n")

        # Ensure datetime is in index
        if 'datetime' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('datetime')

        # Get model components
        model_data = self.trainer.model_data
        trained_model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_columns = model_data['feature_columns']

        # Prepare results storage
        results = []

        # Process each candle progressively
        from tqdm import tqdm
        for i in tqdm(range(len(data)), desc="Processing candles"):
            current_timestamp = data.index[i]

            # Get data UP TO current candle (including current)
            historical_data = data.iloc[:i+1]

            # Extract levels from all timeframes - ONLY data up to current_timestamp
            levels = self._load_levels_from_df(data_dfs, max_date=current_timestamp)

            # Calculate features for current candle using historical data
            all_features = self.trainer.autonomous_trader.get_all_features(
                data=historical_data,
                levels=levels,
                timeframe=timeframe,
                use_log_scale=True,
                current_date=current_timestamp
            )

            # Get features for the current candle (last row)
            current_features = all_features.iloc[-1:]

            # Ensure exact feature match with training features
            for col in feature_columns:
                if col not in current_features.columns:
                    current_features[col] = 0.0

            # Align features with training order
            features_aligned = current_features[feature_columns]

            # Generate prediction for this candle
            probabilities = trained_model.predict_proba(features_aligned)[0]

            # Store results
            result = {
                'timestamp': current_timestamp,
                'close': data.iloc[i]['close']
            }

            # Add probabilities for each class
            classes = label_encoder.classes_
            for j, class_name in enumerate(classes):
                result[f'{class_name}_prob'] = probabilities[j]

            results.append(result)

        # Convert to DataFrame
        signals_df = pd.DataFrame(results)
        signals_df = signals_df.set_index('timestamp')

        # Determine signals based on thresholds
        buy_prob = signals_df.get('buy_prob', pd.Series(0.0, index=signals_df.index))
        sell_prob = signals_df.get('sell_prob', pd.Series(0.0, index=signals_df.index))
        hold_prob = signals_df.get('hold_prob', pd.Series(0.0, index=signals_df.index))

        # Apply thresholds
        signals_df['signal'] = 'HOLD'
        signals_df['confidence'] = hold_prob

        buy_mask = (buy_prob > buy_threshold) & (buy_prob > sell_prob)
        signals_df.loc[buy_mask, 'signal'] = 'BUY'
        signals_df.loc[buy_mask, 'confidence'] = buy_prob[buy_mask]

        sell_mask = (sell_prob > sell_threshold) & (sell_prob > buy_prob)
        signals_df.loc[sell_mask, 'signal'] = 'SELL'
        signals_df.loc[sell_mask, 'confidence'] = sell_prob[sell_mask]

        # Create entry/exit signals for VectorBT
        signals_df['entries'] = (signals_df['signal'] == 'BUY').astype(int)
        signals_df['exits'] = (signals_df['signal'] == 'SELL').astype(int)

        print("\n✅ PROGRESSIVE BACKTESTING COMPLETE")
        print(f"   Processed: {len(signals_df)} candles")
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
