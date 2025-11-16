import pandas as pd
from environments.simple_broker import SimpleBroker
import pytest


def calculate_sl_tp_prices(entry_price, direction, tp_pct, sl_pct):
    """
    Helper to calculate SL/TP prices based on percentages.

    Args:
        entry_price: Entry price
        direction: 1 for LONG, 2 for SHORT
        tp_pct: Take profit percentage (e.g., 0.03 for 3%)
        sl_pct: Stop loss percentage (e.g., 0.02 for 2%)

    Returns:
        (tp_price, sl_price)
    """
    if direction == 1:  # LONG
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SHORT (direction == 2)
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    return tp_price, sl_price


class TestSimpleBroker:
    """Test suite for SimpleBroker class"""

    def setup_method(self):
        """Setup before each test"""
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)
        self.df = pd.DataFrame({
            'open': [100, 102, 104, 106, 108, 110],
            'close': [100, 102, 104, 106, 108, 110],
            'high': [101, 103, 104.8, 107, 109, 111],
            'low': [99, 101, 103, 105, 107, 109]
        })

    def test_initialization(self):
        """Test broker initialization"""
        assert self.broker.initial_balance == 10000.0
        assert self.broker.current_balance == 10000.0
        assert self.broker.equity == 10000.0
        assert self.broker.position_size == 0.0
        assert self.broker.direction == 0
        assert self.broker.total_commission == 0.0

    def test_reset(self):
        """Test reset functionality"""
        # Open and close a position to change state
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)  # Open long
        self.broker.step(1, 0, self.df.iloc[1], tp_price, sl_price)  # Close position

        # Reset and verify
        self.broker.reset()
        assert self.broker.current_balance == 10000.0
        assert self.broker.equity == 10000.0
        assert self.broker.position_size == 0.0
        assert self.broker.realized_pnl == 0.0
        assert self.broker.total_commission == 0.0
        assert len(self.broker.trade_history) == 0

    def test_open_long_position(self):
        """Test opening a long position"""
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)  # signal=1 (LONG)

        assert self.broker.direction == 1
        assert self.broker.position_size > 0
        assert self.broker.avg_entry_price == 100.0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None
        assert self.broker.open_trades == 1
        assert self.broker.traded is True

    def test_open_short_position(self):
        """Test opening a short position"""
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 2, 0.03, 0.02)
        self.broker.step(0, 2, self.df.iloc[0], tp_price, sl_price)  # signal=2 (SHORT)

        assert self.broker.direction == -1
        assert self.broker.position_size < 0
        assert self.broker.avg_entry_price == 100.0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None

    def test_hold_maintains_position(self):
        """Test that HOLD signal (0) maintains position"""

        df = pd.DataFrame({
            'open': [100, 100],
            'close': [100, 101],
            'high': [101, 101],
            'low': [99, 100]
        })

        # Open long position at 100, TP=103, SL=98
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, df.iloc[0], tp_price, sl_price)
        position_size = self.broker.position_size
        direction = self.broker.direction

        # HOLD signal should maintain position (price moves to 101 but doesn't hit TP/SL)
        self.broker.step(1, 0, df.iloc[1], tp_price, sl_price)

        assert self.broker.position_size == position_size  # Position unchanged
        assert self.broker.direction == direction  # Direction unchanged
        assert self.broker.closed_trades == 0  # No trades closed
        assert self.broker.unrealized_pnl > 0  # Should have profit from price move

    def test_stop_loss_long(self):
        """Test stop loss execution for long position"""

        df = pd.DataFrame({
            'open': [100, 100],
            'close': [100, 101],
            'high': [101, 101],
            'low': [99, 98]
        })

        # Open long at 100, SL at 98 (2% below), TP at 104 (4% above)
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.04, 0.02)
        self.broker.step(0, 1, df.iloc[0], tp_price, sl_price)

        # Verify position opened
        assert self.broker.position_size > 0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None

        # Price drops to trigger SL (HOLD signal, but SL triggers)
        self.broker.step(1, 0, df.iloc[1], None, None)

        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'SL'
        assert self.broker.closed_trades == 1

    def test_take_profit_short(self):
        """Test take profit execution for short position"""
        df = pd.DataFrame({
            'open': [100, 100],
            'close': [100, 101],
            'high': [101, 101],
            'low': [99, 96]
        })

        # Open short at 100, TP at 96 (4% below), SL at 102 (2% above)
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 2, 0.04, 0.02)
        self.broker.step(0, 2, df.iloc[0], tp_price, sl_price)

        # Verify position opened
        assert self.broker.position_size < 0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None

        # Price drops to trigger TP (HOLD signal, but TP triggers)
        self.broker.step(1, 0, df.iloc[1], None, None)

        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'TP'
        assert self.broker.closed_trades == 1

    def test_commission_calculation(self):
        """Test commission calculation"""

        # Open position
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)
        commission_open = self.broker.total_commission

        # Close position via direction reversal (open opposite direction)
        tp_price2, sl_price2 = calculate_sl_tp_prices(105.0, 2, 0.03, 0.02)
        self.broker.step(1, 2, self.df.iloc[1], tp_price2, sl_price2)
        commission_total = self.broker.total_commission

        assert commission_open > 0
        assert commission_total > commission_open
        assert self.broker.realized_pnl != 0  # Should have realized PnL from closed long

    def test_unrealized_pnl_calculation(self):
        """Test unrealized PnL calculation"""
        # Open long at 100
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)

        # Check PnL at different prices
        self.broker._update_metrics(105.0)  # Price up 5%
        pnl_up = self.broker.unrealized_pnl
        assert pnl_up > 0

        self.broker._update_metrics(95.0)  # Price down 5%
        pnl_down = self.broker.unrealized_pnl
        assert pnl_down < 0

    def test_equity_calculation(self):
        """Test equity calculation"""
        initial_equity = self.broker.equity

        # Open position
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)

        # Equity should include unrealized PnL
        self.broker._update_metrics(105.0)
        equity_with_profit = self.broker.equity

        assert equity_with_profit > initial_equity

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Execute a winning trade
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)  # Open
        self.broker.step(1, 0, self.df.iloc[1], None, None)  # Close with profit

        perf = self.broker.performance

        assert perf['win_rate'] == 1.0
        assert perf['total_trades'] == 1
        assert perf['total_pnl'] > 0
        assert 'sharpe_ratio' in perf
        assert 'max_drawdown' in perf

    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Initial state
        assert self.broker.max_drawdown == 0.0

        # Simulate a loss
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)  # Open
        self.broker.step(1, 0, self.df.iloc[1], None, None)   # Close with loss

        assert self.broker.max_drawdown > 0.0
        assert self.broker.max_drawdown_value > 0.0

    def test_insufficient_funds(self):
        """Test behavior with insufficient funds"""
        # Create broker with very small balance
        small_broker = SimpleBroker(initial_balance=1.0, maker_commission=0.01)
        self.df = pd.DataFrame({
            'open': [1000, 1000],
            'close': [1000, 1001],
            'high': [1001, 1001],
            'low': [999, 999]
        })
        # Try to open position - should fail gracefully
        tp_price, sl_price = calculate_sl_tp_prices(1000.0, 1, 0.03, 0.02)
        small_broker.step(0, 1, self.df.iloc[0], tp_price, sl_price)

        # Position should not be opened
        assert small_broker.position_size == 0.0
        assert small_broker.open_trades == 0

    def test_step_history(self):
        """Test step history recording"""
        # Execute several steps with HOLD maintaining position

        df = pd.DataFrame({
            'open': [100, 100.5, 101, 101.5, 102],
            'close': [100, 100.5, 101, 101.5, 102],
            'high': [101, 100.5+0.1, 101+0.1, 101.5+0.1, 102+0.1],
            'low': [99, 100.5-0.1, 101-0.1, 101.5-0.1, 102-0.1]
        })

        # TP=103, SL=98, so keep prices between 98-103
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, df.iloc[0], tp_price, sl_price)  # Open long at 100
        # HOLD for several steps (maintains position, prices stay within 99-102.5 range)
        for i in range(1, 5):
            self.broker.step(i, 0, df.iloc[i], tp_price, sl_price)
        assert len(self.broker.step_history) == 5

        for i, step in enumerate(self.broker.step_history):
            assert step['step'] == i
            assert 'equity' in step
            assert 'position_size' in step

        # Position should still be open after HOLDs
        assert self.broker.position_size > 0

    def test_bankruptcy_detection(self):
        """Test bankruptcy detection"""
        # Create broker and lose all money
        broker = SimpleBroker(initial_balance=1000.0, maker_commission=0.001)
        df = pd.DataFrame({
            'open': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'close': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'high': [101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            'low': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89]
        })
        # Simulate massive losses
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.08, 0.08)
        step = 0
        for i in range(5):
            broker.step(step, 1, df.iloc[step], tp_price, sl_price)      # Open
            step += 1
            broker.step(step, 0, df.iloc[step], tp_price, sl_price)      # Close
            step += 1
        # Should detect bankruptcy when equity <= 10% of initial
        assert broker.is_bankrupt == (broker.equity <= broker.initial_balance * 0.1)


def test_edge_cases():
    """Test edge cases and error conditions"""
    broker = SimpleBroker(initial_balance=10000.0)

    # Test with zero price
    with pytest.raises(ValueError):
        broker._calculate_share_size(1000.0, 0.0, 95.0)

    # Test with zero quantity precision
    with pytest.raises(ValueError):
        SimpleBroker(quantity_precision=0.0)

    # Test negative initial balance
    with pytest.raises(ValueError):
        SimpleBroker(initial_balance=-1000.0)
