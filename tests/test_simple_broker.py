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
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # Open long
        self.broker.step(1, 0, 105.0, 106.0, 104.0, tp_price, sl_price)  # Close position

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
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # signal=1 (LONG)

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
        self.broker.step(0, 2, 100.0, 101.0, 99.0, tp_price, sl_price)  # signal=2 (SHORT)

        assert self.broker.direction == -1
        assert self.broker.position_size < 0
        assert self.broker.avg_entry_price == 100.0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None

    def test_hold_maintains_position(self):
        """Test that HOLD signal (0) maintains position"""
        # Open long position at 100, TP=103, SL=98
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        position_size = self.broker.position_size
        direction = self.broker.direction

        # HOLD signal should maintain position (price moves to 101 but doesn't hit TP/SL)
        self.broker.step(1, 0, 101.0, 102.0, 100.5, tp_price, sl_price)

        assert self.broker.position_size == position_size  # Position unchanged
        assert self.broker.direction == direction  # Direction unchanged
        assert self.broker.closed_trades == 0  # No trades closed
        assert self.broker.unrealized_pnl > 0  # Should have profit from price move

    def test_direction_change(self):
        """Test changing from long to short directly"""
        # Open long
        tp_price_long, sl_price_long = calculate_sl_tp_prices(100.0, 1, 0.05, 0.05)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price_long, sl_price_long)

        # Change to short
        tp_price_short, sl_price_short = calculate_sl_tp_prices(102.0, 2, 0.03, 0.02)
        self.broker.step(1, 2, 102.0, 103.0, 101.0, tp_price_short, sl_price_short)

        assert self.broker.direction == -1
        assert self.broker.position_size < 0
        assert self.broker.closed_trades == 1  # Long position closed
        assert self.broker.open_trades == 2    # Short position opened

    def test_stop_loss_long(self):
        """Test stop loss execution for long position"""
        # Open long at 100, SL at 98 (2% below), TP at 104 (4% above)
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.04, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Verify position opened
        assert self.broker.position_size > 0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None

        # Get actual SL/TP prices from broker
        actual_sl = self.broker.stop_loss_price
        actual_tp = self.broker.take_profit_price

        # Price drops to trigger SL (HOLD signal, but SL triggers)
        self.broker.step(1, 0, actual_sl - 0.1, actual_sl + 0.1, actual_sl - 0.1, actual_tp, actual_sl)

        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'SL'
        assert self.broker.closed_trades == 1

    def test_take_profit_short(self):
        """Test take profit execution for short position"""
        # Open short at 100, TP at 96 (4% below), SL at 102 (2% above)
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 2, 0.04, 0.02)
        self.broker.step(0, 2, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Verify position opened
        assert self.broker.position_size < 0
        assert self.broker.stop_loss_price is not None
        assert self.broker.take_profit_price is not None

        # Get actual SL/TP prices from broker
        actual_tp = self.broker.take_profit_price
        actual_sl = self.broker.stop_loss_price

        # Price drops to trigger TP (HOLD signal, but TP triggers)
        self.broker.step(1, 0, actual_tp - 0.1, actual_tp + 0.1, actual_tp - 0.1, actual_tp, actual_sl)

        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'TP'
        assert self.broker.closed_trades == 1

    def test_position_sizing(self):
        """Test position sizing calculation"""
        # Test with different price levels
        broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)

        # High volatility (wide stop) - 5% stop
        tp_price_wide, sl_price_wide = calculate_sl_tp_prices(100.0, 1, 0.03, 0.05)
        broker.step(0, 1, 100.0, 110.0, 90.0, tp_price_wide, sl_price_wide)
        size_wide_stop = abs(broker.position_size)

        broker.reset()

        # Low volatility (tight stop) - 1% stop
        tp_price_tight, sl_price_tight = calculate_sl_tp_prices(100.0, 1, 0.03, 0.01)
        broker.step(0, 1, 100.0, 101.0, 99.0, tp_price_tight, sl_price_tight)
        size_tight_stop = abs(broker.position_size)

        # With tighter stop, position size should be larger (risk per share is smaller)
        assert size_tight_stop > size_wide_stop

    def test_commission_calculation(self):
        """Test commission calculation"""
        initial_balance = self.broker.current_balance

        # Open position
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        commission_open = self.broker.total_commission

        # Close position via direction reversal (open opposite direction)
        tp_price2, sl_price2 = calculate_sl_tp_prices(105.0, 2, 0.03, 0.02)
        self.broker.step(1, 2, 105.0, 106.0, 104.0, tp_price2, sl_price2)
        commission_total = self.broker.total_commission

        assert commission_open > 0
        assert commission_total > commission_open
        assert self.broker.realized_pnl != 0  # Should have realized PnL from closed long

    def test_unrealized_pnl_calculation(self):
        """Test unrealized PnL calculation"""
        # Open long at 100
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

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
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Equity should include unrealized PnL
        self.broker._update_metrics(105.0)
        equity_with_profit = self.broker.equity

        assert equity_with_profit > initial_equity

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Execute a winning trade
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # Open
        self.broker.step(1, 0, 110.0, 111.0, 109.0, tp_price, sl_price)  # Close with profit

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
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # Open
        self.broker.step(1, 0, 90.0, 91.0, 89.0, tp_price, sl_price)   # Close with loss

        assert self.broker.max_drawdown > 0.0
        assert self.broker.max_drawdown_value > 0.0

    def test_insufficient_funds(self):
        """Test behavior with insufficient funds"""
        # Create broker with very small balance
        small_broker = SimpleBroker(initial_balance=1.0, maker_commission=0.01)

        # Try to open position - should fail gracefully
        tp_price, sl_price = calculate_sl_tp_prices(1000.0, 1, 0.03, 0.02)
        small_broker.step(0, 1, 1000.0, 1001.0, 999.0, tp_price, sl_price)

        # Position should not be opened
        assert small_broker.position_size == 0.0
        assert small_broker.open_trades == 0

    def test_trade_history(self):
        """Test trade history recording"""
        # Execute a complete trade by reversing direction
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.05, 0.05)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # Open long

        # Reverse to short (closes long)
        tp_price2, sl_price2 = calculate_sl_tp_prices(103.0, 2, 0.05, 0.05)
        self.broker.step(1, 2, 103.0, 104.0, 102.0, tp_price2, sl_price2)  # Close long, open short

        assert len(self.broker.trade_history) == 2  # Closed long + opened short

        trade = self.broker.trade_history[0]  # The closed long
        assert trade['entry_price'] == 100.0
        assert trade['exit_price'] == 103.0
        assert trade['direction'] == 1
        assert trade['pnl'] > 0
        assert 'duration' in trade
        assert trade['reason'] == 'Direction Change'

    def test_step_history(self):
        """Test step history recording"""
        # Execute several steps with HOLD maintaining position
        # TP=103, SL=98, so keep prices between 98-103
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # Open long at 100

        # HOLD for several steps (maintains position, prices stay within 99-102.5 range)
        for i in range(1, 5):
            price = 100.0 + (i * 0.5)  # 100.5, 101.0, 101.5, 102.0
            self.broker.step(i, 0, price, price + 0.3, price - 0.3, tp_price, sl_price)

        assert len(self.broker.step_history) == 5

        for i, step in enumerate(self.broker.step_history):
            assert step['step'] == i
            assert 'equity' in step
            assert 'position_size' in step

        # Position should still be open after HOLDs
        assert self.broker.position_size > 0

    def test_close_long_position(self):
        """Test CLOSE signal (3) explicitly closes long position"""
        # Open long position at 100, TP at 103, SL at 98
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Verify position opened
        assert self.broker.position_size > 0
        assert self.broker.direction == 1
        assert self.broker.open_trades == 1

        # CLOSE signal (3) at 101 - price between SL (98) and TP (103), won't trigger TP/SL
        self.broker.step(1, 3, 101.0, 102.0, 100.5, None, None)

        # Verify position closed
        assert self.broker.position_size == 0.0
        assert self.broker.direction == 0
        assert self.broker.closed_trades == 1
        assert self.broker.close_reason == 'Manual Close'
        assert self.broker.realized_pnl > 0  # Should have profit

        # Check trade history
        trade = self.broker.trade_history[-1]
        assert trade['status'] == 'CLOSED'
        assert trade['entry_price'] == 100.0
        assert trade['exit_price'] == 101.0
        assert trade['reason'] == 'Manual Close'
        assert trade['pnl'] > 0

    def test_close_short_position(self):
        """Test CLOSE signal (3) explicitly closes short position"""
        # Open short position at 100, TP at 97, SL at 102
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 2, 0.03, 0.02)
        self.broker.step(0, 2, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Verify position opened
        assert self.broker.position_size < 0
        assert self.broker.direction == -1

        # CLOSE signal (3) at 99 - price between TP (97) and SL (102), won't trigger TP/SL
        self.broker.step(1, 3, 99.0, 100.0, 98.5, None, None)

        # Verify position closed
        assert self.broker.position_size == 0.0
        assert self.broker.direction == 0
        assert self.broker.closed_trades == 1
        assert self.broker.close_reason == 'Manual Close'
        assert self.broker.realized_pnl > 0  # Should have profit

        # Check trade history
        trade = self.broker.trade_history[-1]
        assert trade['reason'] == 'Manual Close'

    def test_close_when_no_position(self):
        """Test CLOSE signal (3) when no position is open - should do nothing"""
        initial_balance = self.broker.current_balance

        # Send CLOSE signal when flat
        self.broker.step(0, 3, 100.0, 101.0, 99.0, None, None)

        # Should remain flat and unchanged
        assert self.broker.position_size == 0.0
        assert self.broker.direction == 0
        assert self.broker.current_balance == initial_balance
        assert self.broker.closed_trades == 0
        assert not self.broker.traded

    def test_close_vs_hold_behavior(self):
        """Test difference between HOLD (0) and CLOSE (3) signals"""
        # Open long position
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.05, 0.05)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        position_after_open = self.broker.position_size

        # HOLD (signal 0) should maintain position
        self.broker.step(1, 0, 102.0, 103.0, 101.0, tp_price, sl_price)
        assert self.broker.position_size == position_after_open  # Still open
        assert self.broker.closed_trades == 0

        # CLOSE (signal 3) should close position
        self.broker.step(2, 3, 103.0, 104.0, 102.0, None, None)
        assert self.broker.position_size == 0.0  # Closed
        assert self.broker.closed_trades == 1

        # Subsequent HOLD should keep it flat
        self.broker.step(3, 0, 104.0, 105.0, 103.0, None, None)
        assert self.broker.position_size == 0.0  # Still flat
        assert self.broker.closed_trades == 1  # No new closes

    def test_close_with_loss(self):
        """Test CLOSE signal when position is at a loss"""
        # Open long at 100, TP at 103, SL at 98
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        initial_balance_after_open = self.broker.current_balance

        # CLOSE at 99 - small loss, but price between SL (98) and TP (103)
        self.broker.step(1, 3, 99.0, 99.5, 98.5, None, None)

        # Should have closed with loss
        assert self.broker.position_size == 0.0
        assert self.broker.realized_pnl < 0
        assert self.broker.current_balance < initial_balance_after_open
        assert self.broker.closed_trades == 1

        trade = self.broker.trade_history[-1]
        assert trade['pnl'] < 0
        assert trade['reason'] == 'Manual Close'

    def test_multiple_close_signals(self):
        """Test sending multiple CLOSE signals in sequence"""
        # Open position
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # First CLOSE - should close position
        self.broker.step(1, 3, 105.0, 106.0, 104.0, None, None)
        assert self.broker.position_size == 0.0
        assert self.broker.closed_trades == 1

        # Second CLOSE - should do nothing (already flat)
        self.broker.step(2, 3, 106.0, 107.0, 105.0, None, None)
        assert self.broker.position_size == 0.0
        assert self.broker.closed_trades == 1  # Still 1, no new close

        # Third CLOSE - still should do nothing
        self.broker.step(3, 3, 107.0, 108.0, 106.0, None, None)
        assert self.broker.position_size == 0.0
        assert self.broker.closed_trades == 1

    def test_bankruptcy_detection(self):
        """Test bankruptcy detection"""
        # Create broker and lose all money
        broker = SimpleBroker(initial_balance=1000.0, maker_commission=0.001)

        # Simulate massive losses
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.02)
        for i in range(10):
            broker.step(i * 2, 1, 100.0, 101.0, 99.0, tp_price, sl_price)  # Open
            broker.step(i * 2 + 1, 0, 50.0, 51.0, 49.0, tp_price, sl_price)  # Close with big loss

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
