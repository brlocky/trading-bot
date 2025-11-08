from environments.simple_broker import SimpleBroker
import pytest


def calculate_sl_tp_prices(entry_price, direction, tp_pct, sl_pct):
    """Helper to calculate SL/TP prices based on percentages."""
    if direction == 1:  # LONG
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SHORT (direction == 2)
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    return tp_price, sl_price


class TestTradingSequences:
    """Test specific trading behavior sequences"""

    def setup_method(self):
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)

    def test_same_bet_continuation(self):
        """Test: Model should learn to hold winning positions"""
        # Sequence: [1,2,2] → [1,2,2] → [1,2,2]

        # Open LONG, 5% TP, 3% SL
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.05, 0.03)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        position_size = self.broker.position_size
        sl_price_orig = self.broker.stop_loss_price
        tp_price_orig = self.broker.take_profit_price

        # Keep LONG (price going up)
        tp_price2, sl_price2 = calculate_sl_tp_prices(102.0, 1, 0.03, 0.02)
        self.broker.step(1, 1, 102.0, 103.0, 101.0, tp_price2, sl_price2)
        assert self.broker.position_size == position_size  # Position unchanged
        assert self.broker.stop_loss_price == sl_price_orig     # SL unchanged
        assert self.broker.take_profit_price == tp_price_orig   # TP unchanged

        # Keep LONG (nearing TP)
        self.broker.step(2, 1, 103.5, 104.0, 103.0, tp_price2, sl_price2)
        assert self.broker.position_size == position_size  # Position still unchanged

    def test_early_profit_taking(self):
        """Test: Model should learn when to exit early"""
        # Sequence: [1,2,2] → [1,2,2] → [0,0,0]

        # Open LONG, 3% SL/TP
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Keep LONG (small profit)
        tp_price2, sl_price2 = calculate_sl_tp_prices(102.0, 1, 0.03, 0.03)
        self.broker.step(1, 1, 102.0, 103.0, 101.0, tp_price2, sl_price2)

        # Close manually (take profit)
        self.broker.step(2, 0, 103.0, 104.0, 102.0, tp_price2, sl_price2)

        assert self.broker.position_size == 0.0
        assert self.broker.realized_pnl > 0
        assert self.broker.closed_trades == 1

    def test_direction_switch(self):
        """Test: Model should switch when trend changes"""
        # Sequence: [1,2,2] → [1,2,2] → [2,3,1]
        initial_trades = self.broker.open_trades

        # Open LONG
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        assert self.broker.direction == 1

        # Keep LONG
        tp_price2, sl_price2 = calculate_sl_tp_prices(101.0, 1, 0.03, 0.03)
        self.broker.step(1, 1, 101.0, 102.0, 100.0, tp_price2, sl_price2)
        assert self.broker.direction == 1

        # Switch to SHORT (close LONG + open SHORT)
        tp_price3, sl_price3 = calculate_sl_tp_prices(99.0, 2, 0.04, 0.02)
        self.broker.step(2, 2, 99.0, 100.0, 98.0, tp_price3, sl_price3)
        assert self.broker.direction == -1
        assert self.broker.closed_trades == 1
        assert self.broker.open_trades == initial_trades + 2

    def test_sl_tp_adjustment_ignored(self):
        """Test: SL/TP adjustment attempts should be ignored on hold signals"""
        # Sequence: [1,2,2] → [1,2,1] → [1,3,2]
        # Open LONG, 3% SL/TP
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        original_sl = self.broker.stop_loss_price
        original_tp = self.broker.take_profit_price

        # Try to tighten SL to 2% (should be IGNORED)
        tp_price2, sl_price2 = calculate_sl_tp_prices(101.0, 1, 0.03, 0.02)
        self.broker.step(1, 1, 101.0, 102.0, 100.0, tp_price2, sl_price2)
        assert self.broker.stop_loss_price == original_sl  # SL unchanged
        assert self.broker.take_profit_price == original_tp  # TP unchanged

        # Try to widen TP to 4% (should be IGNORED)
        tp_price3, sl_price3 = calculate_sl_tp_prices(102.0, 1, 0.04, 0.03)
        self.broker.step(2, 1, 102.0, 102.9, 101.0, tp_price3, sl_price3)
        assert self.broker.stop_loss_price == original_sl  # SL unchanged
        assert self.broker.take_profit_price == original_tp  # TP unchanged
